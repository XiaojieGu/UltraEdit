import os

# You can specify the GPU you are using here
os.environ['CUDA_VISIBLE_DEVICES'] =  '0'

from omegaconf import DictConfig

import torch
import torch.nn as nn

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from util import get_module


def make_model(config: DictConfig):
    
    if config.class_name == "AutoModelForFEVER":
        model = AutoModelForFEVER(config.name_or_path)
        model.load_state_dict(torch.load(config.weight_path))
    else:
        model_class = getattr(transformers, config.class_name)
        model = model_class.from_pretrained(config.name_or_path)

    if config.half:
        model.bfloat16()

    for param in model.parameters():
        param.requires_grad = False
        
    for module_name in config.edit_modules:
        print(module_name)
        module = get_module(model, module_name)
        module.weight.requires_grad = True
        
    return model