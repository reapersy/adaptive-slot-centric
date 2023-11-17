import torch
import torch.nn.functional as F
import ipdb
st = ipdb.set_trace


def adjusted_rand_index(true_mask, pred_mask, name='ari_score'):
    r"""Compu