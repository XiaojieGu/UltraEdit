from typing import Union, Tuple, List, Dict
from omegaconf import DictConfig
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers.pytorch_utils import Conv1D
import nltk
from transformers import AutoModelForCausalLM, AutoTokenizer
import unicodedata
import typing,json
from itertools import chain


def empty_cache(path: str, config):

    dir_path = f"{config.editor.cache_dir}/{config.model.name}_{config.dataset.name}_{config.editor.name}_{config.dataset.n_edits}_{config.num_seq}"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    try:
        for file_name in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"Error while clearing cache: {e}")


def get_module(module: nn.Module, module_name: str) -> nn.Module:
    
    for name in module_name.split("."):
        module = getattr(module, name)
    return module


def get_shape(module: Union[nn.Linear, Conv1D]) -> Tuple[int]:
    
    shape = tuple(module.weight.shape)
    return shape[::-1] if isinstance(module, nn.Linear) else shape
    
    
def cross_entropy(
    logits: torch.FloatTensor,
    labels: torch.LongTensor
):
    if len(logits.shape) == 2:
        return F.binary_cross_entropy_with_logits(logits, labels)

    if len(logits.shape) == 3:
        ans_indice = torch.where(labels != -100)
        logits = logits[ans_indice]
        labels = labels[ans_indice]
        
        return F.cross_entropy(logits, labels)


def log(x: torch.FloatTensor) -> torch.FloatTensor:
    return (x + torch.finfo(x.dtype).eps).log()


def kl_div(
    refer_logits: torch.FloatTensor,
    logits: torch.FloatTensor,
    labels: torch.LongTensor
) -> torch.Tensor:
    
    if len(logits.shape) == 2:
        refer_probs = F.sigmoid(refer_logits)
        probs = F.sigmoid(logits)
        return (refer_probs * (log(refer_probs) - log(probs))) + ((1 - refer_probs) * (log(1 - refer_probs) - log(1 - probs)))
    
    if len(logits.shape) == 3:
        ans_indice = torch.where(labels != -100)
        refer_logits = refer_logits[ans_indice]
        logits = logits[ans_indice]
        refer_log_probs = refer_logits.log_softmax(-1)
        log_probs = logits.log_softmax(-1)
        
        return F.kl_div(
            log_probs,
            refer_log_probs,
            reduction = "batchmean",
            log_target = True
        )


def succ_ratios(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    old_labels: torch.LongTensor=None
) -> List[float]:
    
    if old_labels is None:
    
        if len(logits.shape) == 2:
            return ((logits > 0) == labels).squeeze(-1).to("cpu").numpy().tolist()
        
        if len(logits.shape) == 3:
            n_corr = (logits.argmax(-1) == labels).sum(-1)
            n_tokens = (labels != -100).sum(-1)
            return (n_corr / n_tokens).to("cpu").numpy().tolist()
    
    else:

        if len(logits.shape) == 2:

            if old_labels.shape[1] > labels.shape[1]:
                old_labels = old_labels[:, :labels.shape[1]]
            label_probs = logits[torch.arange(logits.size(0)), labels]
            old_label_probs = logits[torch.arange(logits.size(0)), old_labels]
            success = (label_probs > old_label_probs).to(torch.float32)

        if len(logits.shape) == 3:
            # import ipdb;ipdb.set_trace()
            batch_size, seq_len, _ = logits.shape

            if old_labels.shape[1] > labels.shape[1]:
                old_labels = old_labels[:, :labels.shape[1]]

            if labels.shape[1] > old_labels.shape[1]:
                move = labels.shape[1] - old_labels.shape[1]
                labels = labels[:, :old_labels.shape[1]]
                seq_len -= move

            valid_mask = (labels != -100) & (old_labels != -100)
            label_probs = logits[torch.arange(batch_size).unsqueeze(1), torch.arange(seq_len), labels]
            old_label_probs = logits[torch.arange(batch_size).unsqueeze(1), torch.arange(seq_len), old_labels]
            success = ((label_probs > old_label_probs) & valid_mask).to(torch.float32)

        n_corr = success.sum(-1)
        n_tokens = (labels != -100).sum(-1)

        return (n_corr / n_tokens).to("cpu").numpy().tolist()




class Tracer:

    def __init__(
        self,
        module: nn.Module,
        cache_mask: torch.LongTensor
    ):
        cache_indices = torch.where(cache_mask)

        def forward_hook(
            module: nn.Module,
            inputs: Tuple[torch.FloatTensor],
            outputs: Tuple[torch.FloatTensor]
        ):
            self.keys = inputs[0][cache_indices].detach()
            
        def backward_hook(
            module: nn.Module,
            inputs_grad: Tuple[torch.FloatTensor],
            outputs_grad: Tuple[torch.FloatTensor]
        ):
            self.values_grad = outputs_grad[0][cache_indices].detach()

        self.handles = [
            module.register_forward_hook(forward_hook),
            module.register_full_backward_hook(backward_hook)
        ]





class TracerDict(dict):
    
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        tuples: Dict[str, torch.LongTensor]
    ):
        
        if any("encoder" in m for m in config.model.edit_modules) and any("decoder" in m for m in config.model.edit_modules):
            
            for module_name in config.model.edit_modules:
                if "encoder" in module_name:
                    cache_mask = tuples["attention_mask"]
                else:
                    cache_mask = tuples["decoder_attention_mask"]
                module = get_module(model, module_name)
                self[module_name] = Tracer(module, cache_mask)

        else:

            if config.editor.token == "ans":
                cache_mask = tuples["labels"] != -100
            else:
                cache_mask = tuples["attention_mask"]

            for module_name in config.model.edit_modules:
                module = get_module(model, module_name)
                self[module_name] = Tracer(module, cache_mask)
            
    def __enter__(self):
        return self
            
    def __exit__(self, type, value, traceback):
        for v in self.values():
            for h in v.handles:
                h.remove()