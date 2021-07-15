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

class ClevrDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        root_file = opt.root_folder
        self.all_files = glob.glob(f'{root_file}/clevr_train/*')

        self.resize = torchvision.transforms.Resize((opt.image_height,opt.image_width))
        self.resize_mask = torchvision.transforms.Resize((opt.image_height,opt.image_width),torchvision.transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        if self.opt.overfit:
            idx = 0

        file_val = self.all_files[idx]
        pickled_file = pickle.load(open(file_val,'rb'))
        
        rgb_val = torch.from_numpy(pickled_file['image']).squeeze().float()
        images = rgb_val / 256.0 
        # Normalize to [0, 1] range.
        gt_mask_val = torch.from_numpy(np.argmax(pickled_file['mask'].squeeze(),0))
        
        images = images.permute(2,0,1).unsqueeze(0)
        images = self.resize(images)
        gt_mask_val = self.resize_mask(gt_ma