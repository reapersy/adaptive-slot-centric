import models
import os
import torch 
import ipdb
st = ipdb.set_trace

def get_model_and_optimizer(opt):
    model = models.ModelIter(opt)
    model = model.to(opt.device)
    
    if opt.tta_optimizer == "adam":
   