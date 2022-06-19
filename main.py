
import hydra
from tqdm import tqdm
import time
import ipdb 
import matplotlib.pyplot as plt
st = ipdb.set_trace
import torch
import model_utils
import wandb
import random
import utils
import os
import numpy as np
from dataset import ClevrDataset
import dataset
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from omegaconf import open_dict
from omegaconf import OmegaConf

def parse_args(opt):
    with open_dict(opt):
        opt.log_dir = os.getcwd()
        print(f"Logging files in {opt.log_dir}")
        opt.device = "cuda:0" if opt.use_cuda else "cpu"
        opt.cwd = get_original_cwd()

    if not opt.use_random_seed:
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        random.seed(opt.seed)
    
    if opt.deep_tta_vis: