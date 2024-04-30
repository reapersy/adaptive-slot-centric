
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import ipdb 
st = ipdb.set_trace


def get_learning_rate(opt, step):
    if opt.learning_rate_schedule == 0:
        return opt.learning_rate
    elif opt.learning_rate_schedule == 1:
        return get_linear_warmup_lr(opt, step)
    else:
        raise NotImplementedError


def get_linear_warmup_lr(opt, step):
    if step < opt.warmup_steps:
        return opt.learning_rate * step / opt.warmup_steps
    else:
        return opt.learning_rate

def update_learning_rate(optimizer, opt, step):
    lr = get_learning_rate(opt, step)
    optimizer.param_groups[0]["lr"] = lr
    return optimizer, lr


def summ_instance_masks(masks,  pred=False):
    masks = masks.squeeze(1)
    if pred:
        old_shape = masks.shape
        num_slots = masks.shape[0]