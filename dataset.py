import pickle
import socket
from PIL import Image
import glob
import torchvision
import numpy as np
import ipdb
import torch.nn.functional as F
st = ipdb.set_trace
import torch
import time

class ClevrDataset(torch.utils.d