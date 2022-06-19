
import hydra
from tqdm import tqdm
import time
import ipdb 
import matplotlib.pyplot as plt
st = ipdb.set_trace
import torch
import model_utils
import wandb
import random
import utils
import os
import numpy as np
from dataset import ClevrDataset
import dataset
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from omegaconf import open_dict
from omegaconf import OmegaConf

def parse_args(opt):
    with open_dict(opt):
        opt.log_dir = os.getcwd()
        print(f"Logging files in {opt.log_dir}")
        opt.device = "cuda:0" if opt.use_cuda else "cpu"
        opt.cwd = get_original_cwd()

    if not opt.use_random_seed:
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        random.seed(opt.seed)
    
    if opt.deep_tta_vis:
        opt.log_freq = 1
    
    print(OmegaConf.to_yaml(opt))
    return opt


def do_tta(opt, model, optimizer, tta_dataset):
    model.eval()
    step =0
    before_tta_acc = []
    after_tta_acc = []
    run_name = wandb.run.name
    
        
    


    before_tta_acc_fg = []
    after_tta_acc_fg = []
    for index_val in tqdm(range(0,len(tta_dataset))):
        all_losses = []
        all_accs = []
        
        if opt.deep_tta_vis:
            folder_name = f"tta_dump/{run_name}/{index_val}"
            gt_rgb_folder_name = f"tta_dump/{run_name}/{index_val}/gt_rgb"
            pred_mask_folder_name = f"tta_dump/{run_name}/{index_val}/pred_mask"
            pred_rgb_folder_name = f"tta_dump/{run_name}/{index_val}/pred_rgb"
            os.makedirs(folder_name, exist_ok=True)
            os.makedirs(gt_rgb_folder_name, exist_ok=True)
            os.makedirs(pred_mask_folder_name, exist_ok=True)
            os.makedirs(pred_rgb_folder_name, exist_ok=True)

        for tta_step in tqdm(range(opt.tta_steps)):
            images, gt_mask_val , gt_indices = tta_dataset[index_val]
            images, gt_mask_val , gt_indices = (images.unsqueeze(0).to(opt.device),gt_mask_val.unsqueeze(0).to(opt.device),gt_indices.unsqueeze(0).to(opt.device))    
            feed_dict = {}
            feed_dict["image"] = images
            feed_dict["gt_mask"] = gt_mask_val
            feed_dict["gt_indices"] = gt_indices

            if tta_step ==0:
                with torch.no_grad():
                    model.eval()
                    loss, vis_dict = model(feed_dict, step)
                    before_tta_acc.append(vis_dict["ari_score"])
                    before_tta_acc_fg.append(vis_dict["fg_ari_score"])            
            
            learning_rate = optimizer.param_groups[0]['lr']
            feed_dict["learning_rate"] = learning_rate
            
            loss, vis_dict = model(feed_dict, step)

            if opt.deep_tta_vis:
                vis_dict['gt_rgb'].image.save(f"{gt_rgb_folder_name}/{tta_step:04d}.png")
                vis_dict['pred_mask'].image.save(f"{pred_mask_folder_name}/{tta_step:04d}.png")
                vis_dict['pred_rgb'].image.save(f"{pred_rgb_folder_name}/{tta_step:04d}.png")
            
            all_losses.append(vis_dict['reconstruction_loss'])
            all_accs.append(vis_dict['ari_score'])



            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            vis_dict["learning_rate"] = learning_rate
            
            step += 1

            if tta_step == opt.tta_steps-1:
                with torch.no_grad():
                    model.eval()
                    loss, vis_dict = model(feed_dict, step)
                    after_tta_acc.append(vis_dict["ari_score"])
                    after_tta_acc_fg.append(vis_dict["fg_ari_score"])                
                    vis_dict["before_tta_mean_acc"] = np.array(before_tta_acc).mean()
                    vis_dict["before_tta_mean_acc_fg"] = np.array(before_tta_acc_fg).mean()                                
                    vis_dict["after_tta_mean_acc"] = np.array(after_tta_acc).mean()
                    vis_dict["after_tta_mean_acc_fg"] = np.array(after_tta_acc_fg).mean()                                