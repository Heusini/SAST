import os
import ast
import random
import argparse
import subprocess
from pathlib import Path

import hydra
import torch
import numpy as np
from PIL import Image

from modules.utils.fetch import fetch_model_module
from utils.timers import CudaTimer as CudaTimer
from omegaconf import OmegaConf, DictConfig
import torch
from benchmark import compute_gflops
from thop import profile
from thop import clever_format


def gflops(model):
    model.eval()
    model = model.cuda()

    dummy_input = torch.randn(1, 3, 384, 640).cuda()
    rgb_image = torch.randn(1, 20, 384, 640).cuda()
    macs, params = profile(model, inputs=(dummy_input,rgb_image,), verbose=False)
    flops = 2 * macs

    flops_formatted, _ = clever_format([flops, params], "%.3f")

    print(f"FLOPs: {flops_formatted}")
    print(f"Parameters: {params}")


@hydra.main(config_path='config', config_name='detect', version_base='1.2')
def main(config: DictConfig):
     # Load the configuration file
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    model_name = config.model.name
    

    # device for export onnx
    device = torch.device(f"cuda:0")
    
    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

     # Load the model from the checkpoint
    module = fetch_model_module(config=config)
    ckpt_path = config.checkpoint
    # if ckpt_path is not None and config.wandb.wandb.resume_only_weights:
    if ckpt_path:
        print('Resuming only the weights instead of the full training state')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt["state_dict"]

        backbone_str = "mdl.backbone."
        backbone_dict = {k.replace(backbone_str, ""): v 
                     for k, v in state_dict.items() if k.startswith(backbone_str)}

        fpn_str = "mdl.fpn."
        fpn_dict = {k.replace(fpn_str, ""): v 
                     for k, v in state_dict.items() if k.startswith(fpn_str)}

        module.mdl.backbone.load_state_dict(backbone_dict, strict=True)
        module.mdl.fpn.load_state_dict(fpn_dict, strict=True)
    module.to(device)

    # Make sure the model is in evaluation mode
    module.eval()

    # compute_gflops(module, None, True, 0.0)
    gflops(module)



if __name__ == '__main__':
    main()
