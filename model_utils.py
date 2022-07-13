import models
import os
import torch 
import ipdb
st = ipdb.set_trace

def get_model_and_optimizer(opt):
    model = models.ModelIter(opt)
    model = model.to(opt.device)
    
    if opt.tta_optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, momentum=0.9)
    
    if opt.load_folder != "None":
        print("Loading model from", opt.load_folder)
        # st()
        load_path = opt.cwd + "/" + opt.load_folder
        state_dict = torch.load(load_path)
        model.load_state_dict(state_dict["model"])    
        if opt.tta_optimizer == "adam":
 