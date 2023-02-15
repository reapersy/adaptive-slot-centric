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


        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        all_attn_slot = []
        all_attn = []


        for iter_num in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale

            attn = dots.softmax(dim=1) + self.eps
            attn_slot = attn
            attn = attn / attn.sum(dim=-1, keepdim=True)
            
            all_attn.append(attn)

            all_attn_slot.append(attn_slot)

            updates = torch.einsum('bjd,bij->bid', v, attn)                
            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
        return slots, all_attn, all_attn_slot


"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid_encoder(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid


class Encoder(nn.Module):
    def __init__(self, resolution, hid_dim, in_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, hid_dim, 5, padding = 2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)              
        self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)            
        x = x.permute(0,2,3,1)
        x = self.encoder_pos(x)
        return x


class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.

    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super(ResnetBlockFC, self).__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out
        # Submodules
        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        # Initialization
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx




class ImplicitMLP2DDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''
    def __init__(self, dim=2, c_dim=64,
                 hidden_size=32, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1, out_dim=1, grid_there = False, resolution=None):
        super(ImplicitMLP2DDecoder, self).__init__()
        print('Implicit Local Decoder...')
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.hidden_size = hidden_size
        self.xyz_grid = self.build_grid2D_imp(resolution)
        self.xyz_grid = self.xyz_grid*(resolution[0]-1 )
        self.fc_p = nn.Linear(dim, hidden_size)
        self.resolution = resolution

        self.fc_c = nn.ModuleList([
            nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
        ])

        self.blocks = nn.ModuleList([ResnetBlockFC(hidden_size) for i in range(n_blocks)])


        self.fc_out = nn.Linear(hidden_size, out_dim)

        self.out_dim = out_dim
        
        self.actvn = F.relu

        self.padding = padding



    def build_grid2D_imp(self,resolution):
        ranges = [np.linspace(0., 1., num=res) f