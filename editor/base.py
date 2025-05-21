from typing import Dict, List
from omegaconf import DictConfig

from collections import Counter
import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import islice

from tqdm import tqdm
import wandb

from transformers import AutoTokenizer

import json

from model import make_model
from util import (
    get_module,
    get_shape,
    empty_cache,
    TracerDict,
    cross_entropy,
    kl_div,
    succ_ratios
)


class BaseEditor:

    def __init__(
        self,
        config: DictConfig,
        model: nn.Module
    ):
        
        self.config = config
        self.model = model
        
        shape_counter = Counter()
        self.name2idx = {}
        for module_name in config.model.edit_modules:
            shape = get_shape(get_module(model, module_name))
            self.name2idx[module_name] = shape_counter[shape]
            shape_counter[shape] += 1
        
        self.shape_counter = shape_counter

        self.tuples_list = []
        if 'llama' in config.model.name_or_path:
            self.tok= LlamaTokenizerFast.from_pretrained(config.model.name_or_path)
        else:
            self.tok = AutoTokenizer.from_pretrained(config.model.name_or_path)


    def edit_model(
        self,
        param_shifts: Dict[str, torch.FloatTensor],
        is_reverse: bool
    ):
        
        for module_name, param_shift in param_shifts.items():
            module = get_module(self.model, module_name)
            if isinstance(module, nn.Linear):
                param_shift = param_shift.T
            if is_reverse:
                param_shift = - param_shift
            module.weight.data += param_shift.to(module.weight.data.dtype)


    def reset_model(self):
        del self.model
        torch.cuda.empty_cache()
        self.model = make_model(self.config.model).to(self.config.model_device)


    def cache(self, tuples: List[Dict[str, torch.LongTensor]]):

        for idx, t in enumerate(tuples):
            

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
                self.net[str(shape)].normalizer.update(torch.cat((keys, values_grad), -1))
                dir_path = f"{self.config.editor.cache_dir}/{self.config.model.name}_{self.config.dataset.name}_{self.config.editor.name}_{self.config.dataset.n_edits}_{self.config.num_seq}"
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path,exist_ok=True)
                torch.save(keys, f"{dir_path}/{module_idx}_{idx}_keys.pth")
                torch.save(values_grad, f"{dir_path}/{module_idx}_{idx}_values_grad.pth")


    def train(self, loader: DataLoader, save=False):
        """
        training method for MEND and MALMEN.
        """
        max_steps = self.config.num_seq

        limited_loader = loader

        for _, tuples in enumerate(tqdm(limited_loader, desc = "Train", ncols = 100,total=max_steps)):

            self.cache(tuples["edit_tuples"])
            param_shifts = self.predict_param_shifts()
            self.model.zero_grad()

            gen_losses = []
            self.edit_model(param_shifts, False)
            for t in tuples["equiv_tuples"]:
                logits = self.model(**t)["logits"]
                loss = cross_entropy(logits, t["labels"])
                loss.backward()
                gen_losses += [loss.item()]
            self.edit_model(param_shifts, True)

            loc_losses = []
            for t in tuples["unrel_tuples"]:


                with torch.no_grad():
                    refer_logits = self.model(**t)["logits"]

                self.edit_model(param_shifts, False)
                logits = self.model(**t)["logits"]

                loss = kl_div(
                    refer_logits,
                    logits,
                    t["labels"]
                )
                (self.config.editor.loc_coef * loss).backward()
                self.edit_model(param_shifts, True)
                loc_losses += [loss.item()]
                self.update_hypernet(param_shifts, update=True)

            wandb.log({
                "gen_loss": np.mean(gen_losses),
                "loc_loss": np.mean(loc_losses)
            })
        
        if save:
            torch.save(self.net, "hypernet.pt")


    def sequential_valid(self, loader: DataLoader):
        """
        Valid the entire knowledge sequence, with just final results showed.
        """

        max_steps = self.config.num_seq
        limited_loader = islice(loader, max_steps)

        for _, tuples in enumerate(tqdm(limited_loader, desc="Editing Time", ncols=100, total=max_steps)):

            self.cache(tuples["edit_tuples"])
            param_shifts = self.predict_param_shifts()
                
            self.edit_model(param_shifts, False)
            self.tuples_list.append(tuples)
            if self.config.editor.name != "ultraedit":
                self.opt.zero_grad()

        edit_succs, gen_succs, loc_succs = [], [], []
        for k, s in zip(
            ["edit_tuples", "equiv_tuples", "unrel_tuples"],
            [edit_succs, gen_succs, loc_succs]
        ):
            for tuple in tqdm(self.tuples_list, desc=f"Eval time of {k}", ncols=100,total=len(self.tuples_list)):
                for t in tuple[k]:
                    with torch.no_grad():
                        logits = self.model(**t)["logits"]
                    s += succ_ratios(logits, t["labels"])
        if self.config.model_cache==True:
            if not os.path.exists(self.config.model_cache_dir):
                os.makedirs(self.config.model_cache_dir)
            self.model.save_pretrained(self.config.model_cache_dir)
            self.tok.save_pretrained(self.config.model_cache_dir)
        
        if self.config.dataset.name=="wikibigedit":
            person_succs=[]
            mhop_succs=[]
            if self.config.dataset.eval_mhop==True:
                for k, s in zip(
                    ["person_tuples", "mhop_tuples"],
                    [person_succs, mhop_succs]
                ):
                    for tuple in tqdm(self.tuples_list, desc=f"Eval time of {k} ", ncols=100,total=len(self.tuples_list)):
                        for t in tuple[k]:
                            # import ipdb;ipdb.set_trace()
                            with torch.no_grad():
                                logits = self.model(**t)["logits"]
                            s += succ_ratios(logits, t["labels"])
            else:
                for k, s in zip(
                    ["person_tuples"],
                    [person_succs]
                ):
                    for tuple in tqdm(self.tuples_list, desc=f"Eval time of {k}", ncols=100,total=len(self.tuples_list)):
                        for t in tuple[k]:
                            # import ipdb;ipdb.set_trace()
                            with torch.no_grad():
                                logits = self.model(**t)["logits"]
                            s += succ_ratios(logits, t["labels"])

                    
        final_results={
            "ES": np.mean(edit_succs),
            "GS": np.mean(gen_succs),
            "LS": np.mean(loc_succs)
        }
        if self.config.dataset.name=="wikibigedit":
            final_results["person_score"]=np.mean(person_succs)
            if self.config.dataset.eval_mhop==True:
                final_results["mhop_score"]=np.mean(mhop_succs)

        print("final_results_sequential:")
        print(final_results)

        wandb.log(final_results)

    def run(self, train_loader: DataLoader, valid_loader: DataLoader):
        """
        Use MEND or MALMEN to complete sequential editing task.
        Just train the hypernet on the original LLM, then freeze it.
        """

        for _ in range(self.config.editor.n_epochs):

            self.train(train_loader)
            self.reset_model()
            self.sequential_valid(valid_loader)
            empty_cache(self.config.editor.cache_dir, self.config)
            self.reset_hypernet()





