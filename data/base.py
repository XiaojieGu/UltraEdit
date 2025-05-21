from typing import Union, Tuple, List, Dict
from omegaconf import DictConfig

import math
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer



class BaseDataset(Dataset):

    def __init__(
        self,
        config: DictConfig,
        path: str,
        tok: AutoTokenizer,
        device: Union[int, str, torch.device],
        clip=True
    ):
        self.config = config
        with open(path) as file:
            self.data = json.load(file)
            if clip==True:
                self.data=self.data[:self.config.n_edits*self.config.num_seq]
            else:
                self.data=self.data
                
        self.tok = tok
        self.tok.pad_token=self.tok.eos_token
        self.device = device

    def __len__(self):
        return len(self.data)


    def collate_fn(
        self,
        tuples: Tuple[Dict[str, Dict[str, torch.LongTensor]]]
    ) -> Dict[str, List[Dict[str, torch.LongTensor]]]:
        tuples: Dict[str, List[Dict[str, torch.LongTensor]]] = {
            k: sorted(
                [t[k] for t in tuples],
                key = lambda x: x["attention_mask"].sum().item(),
                reverse = True
            )
            for k in tuples[0].keys()
        }
        
        return {
            k: [
                self.pad_tok_tuples(v[n_batch * self.config.batch_size:(n_batch + 1) * self.config.batch_size])
                for n_batch in range(math.ceil(self.config.n_edits / self.config.batch_size))
            ]
            for k, v in tuples.items()
        }
        

    def pad_tok_tuples(
        self,
        tok_tuples: List[Dict[str, torch.LongTensor]]
    ) -> Dict[str, torch.LongTensor]:
        
        return {
            k: pad_sequence(
                [t[k].squeeze(0) for t in tok_tuples],
                batch_first = True,
                padding_value = -100 if k == "labels" else 0
            ).to(self.device)
            for k in tok_tuples[0].keys()
        }



def make_loader(
    config: DictConfig,
    data_class
) -> Tuple[DataLoader]:

    tok = AutoTokenizer.from_pretrained(config.model.name_or_path)

    if config.editor.name!='ultraedit':
        if (config.editor.name=='mend' or config.editor.name=='malmen') and (config.dataset.name=='fever' or config.dataset.name=='zsre'):
            train_set = data_class(
                config.dataset,
                config.dataset.train_path,
                tok,
                config.model_device,
                clip=False
            )
        else:
            train_set = data_class(
                config.dataset,
                config.dataset.train_path,
                tok,
                config.model_device
            )

        train_loader = DataLoader(
            train_set,
            config.dataset.n_edits,
            False,
            collate_fn = train_set.collate_fn,
            drop_last = True
        )

    valid_set = data_class(
        config.dataset,
        config.dataset.valid_path,
        tok,
        config.model_device
    )


    valid_loader = DataLoader(
        valid_set,
        config.dataset.n_edits,
        False,
        collate_fn = valid_set.collate_fn,
        drop_last = True
    )

    if config.editor.name!='ultraedit':
        return train_loader, valid_loader
    else:
        return None, valid_loader
    