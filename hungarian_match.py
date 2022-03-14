from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import ipdb
st = ipdb.set_trace

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Beca