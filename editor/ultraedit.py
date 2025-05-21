from typing import Dict
from omegaconf import DictConfig
import math
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from editor.base import BaseEditor
from util import get_module, get_shape
import numpy as np
from nets import RLEditNet, RunningMeanStd
from itertools import islice
from tqdm import tqdm
import os
import wandb
from typing import Dict, List

from util import (
    get_module,
    get_shape,
    empty_cache,
    cross_entropy,
    kl_div,
    TracerDict,
)
def pad_tensor(tensor, target_length, dim=0, padding_value=0):

    tensor_length = tensor.size(dim)
    if tensor_length >= target_length:
        return tensor.narrow(dim, 0, target_length)
    else:
        padding = target_length - tensor_length
        pad_shape = list(tensor.shape)
        pad_shape[dim] = padding
        pad_tensor = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
        mask = torch.cat([torch.ones(tensor_length, dtype=torch.float32, device=tensor.device),
                          torch.zeros(padding, dtype=torch.float32, device=tensor.device)], dim=0)
        return torch.cat([tensor, pad_tensor], dim=dim)


class ULTRAEDIT(BaseEditor):

    def __init__(
        self,
        config: DictConfig,
        model: nn.Module
    ):
        super().__init__(
            config,
            model
        )

        self.lifelong_normalizer = nn.ModuleDict({
            str(k): RunningMeanStd(
                k[0]+k[1],
            )
            for k, v in self.shape_counter.items()
        }).to(config.editor_device)

    def predict_param_shifts(self) -> Dict[str, torch.FloatTensor]:

        param_shifts = {}
        for module_idx, module_name in enumerate(self.config.model.edit_modules):

            shape = get_shape(get_module(self.model, module_name))

            lifelong_normalizer = self.lifelong_normalizer[str(shape)]
    
            hidden_states = torch.cat([
                torch.load(f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.dataset.name}_{self.config.editor.name}_{self.config.dataset.n_edits}_{self.config.num_seq}/{module_idx}_{idx}_keys.pth")
                for idx in range(math.ceil(self.config.dataset.n_edits / self.config.dataset.batch_size))
            ])
            values_grad = torch.cat([
                torch.load(f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.dataset.name}_{self.config.editor.name}_{self.config.dataset.n_edits}_{self.config.num_seq}/{module_idx}_{idx}_values_grad.pth")
                for idx in range(math.ceil(self.config.dataset.n_edits // self.config.dataset.batch_size))
            ])

            v_feature = torch.empty((0, shape[1]), device = self.config.editor_device)
            for start_idx in range(0, hidden_states.shape[0], self.config.editor.batch_size):
                end_idx = start_idx + self.config.editor.batch_size
                hidden_states_once = pad_tensor(hidden_states[start_idx:end_idx], self.config.editor.batch_size, 0)
                values_grad_once = pad_tensor(values_grad[start_idx:end_idx], self.config.editor.batch_size, 0)
                with torch.no_grad():
                    z_feature = torch.cat((hidden_states_once, values_grad_once), -1)

                    z_feature = lifelong_normalizer(z_feature)
                    (hidden_states_hat, pesudo_values_hat) = z_feature.split([shape[0], shape[1]], -1)
                
                    coeffs = - self.config.editor.lr*(hidden_states_hat * hidden_states_hat).sum(-1).unsqueeze(-1)
                v_feature = torch.cat((v_feature, coeffs * pesudo_values_hat))
            with torch.no_grad():
                mat = hidden_states.T @ hidden_states + torch.eye(shape[0], device=self.config.editor_device)
            v_feature = v_feature[:hidden_states.shape[0], :]
            param_shift = torch.linalg.solve(mat, hidden_states.T @ v_feature)
            param_shifts[module_name] = param_shift.to(next(self.model.parameters()).device)
            
        return param_shifts



    def cache(self, tuples: List[Dict[str, torch.LongTensor]]):

        for idx, t in enumerate(tuples):
            
            if "old_labels" in t:
                old_labels = t.pop("old_labels")

            with TracerDict(
                self.model,
                self.config,
                t
            ) as tr:
                logits = self.model(**t)["logits"]
                cross_entropy(logits, t["labels"]).backward()
        
            for module_idx, module_name in enumerate(self.config.model.edit_modules):
                shape = get_shape(get_module(self.model, module_name))
                keys = tr[module_name].keys.to(torch.float32).to(self.config.editor_device)
                values_grad = tr[module_name].values_grad.to(torch.float32).to(self.config.editor_device)
                self.lifelong_normalizer[str(shape)].update(torch.cat((keys, values_grad), -1))
                dir_path = f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.dataset.name}_{self.config.editor.name}_{self.config.dataset.n_edits}_{self.config.num_seq}"
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path,exist_ok=True)
                torch.save(keys, f"{dir_path}/{module_idx}_{idx}_keys.pth")
                torch.save(values_grad, f"{dir_path}/{module_idx}_{idx}_values_grad.pth")

            try:
                t["old_labels"] = old_labels
            except:
                pass





    def run(self, train_loader: DataLoader, valid_loader: DataLoader):
        """
        Use UltraEdit to finish sequential editing task.
        """
        
        self.sequential_valid(valid_loader)
        self.reset_model()
        empty_cache(self.config.editor.cache_dir, self.config)
        self.reset_model()



























