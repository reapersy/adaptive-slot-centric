import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import numpy as np
import wandb
import segmentation_metric
from hungarian_match import HungarianMatcher
import time
import utils
import ipdb
st = ipdb.set_trace


def build_grid_encoder(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], 