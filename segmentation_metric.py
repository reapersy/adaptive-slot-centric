import torch
import torch.nn.functional as F
import ipdb
st = ipdb.set_trace


def adjusted_rand_index(true_mask, pred_mask, name='ari_score'):
    r"""Computes the adjusted Rand index (ARI), a clustering similarity score.
    This implementation ignores points with no cluster label in `true_mask` (i.e.
    those points for which `true_mask` is a zero vector). In the context of
    segmentation, that means this