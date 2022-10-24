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
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).cuda()



class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128, pos_dims=0):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))

        self.feat_dim = dim
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)


        self.to_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        hidden_dim = max(dim, hidden_dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)

        self.norm_input  = nn.LayerNorm(dim)




    def forward(self, inputs):
        b, n, d = inputs.shape        
        
        slots = self.slots_mu.repeat([b,1,1])


        inputs = self.norm_input(