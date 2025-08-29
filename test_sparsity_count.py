import os
from pathlib import Path
import torch as th
import torch.nn as nn

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')

import cv2
import sys
import hydra
import numpy as np
import bbox_visualizer as bbv
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
# from pytorch_lightning.callbacks import ModelSummary

from pytorch_lightning.utilities.model_summary import ModelSummary

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_data_module, fetch_model_module
from data.utils.types import DataType, LstmStates, ObjDetOutput, DatasetSamplingMode, BackboneFeatures
from utils.padding import InputPadderFromShape

import matplotlib.pyplot as plt
from benchmark import measure_average_inference_time

PATH = Path("/datasets/sheusinger/st_stephan_360_640_20/")

def get_sparsity_mask(fpn_layer: th.Tensor, threshold = 0.7):
    max_pool = nn.MaxPool2d(2, 2)
    tokens = fpn_layer
    tokens = th.norm(tokens, dim=1)
    min_val = tokens.amin(dim=(-2, -1), keepdim=True)
    max_val = tokens.amax(dim=(-2, -1), keepdim=True)

    tokens = (tokens - min_val) / (max_val-min_val + 1e-8)
    if max_pool is not None:
        tokens = max_pool(tokens)
    #     print("max_pool")

    return tokens

@hydra.main(config_path='config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=False)
    gpus = config.hardware.gpus
    assert isinstance(gpus, int), 'no more than 1 GPU supported'
    gpus = [gpus]

    # ---------------------
    # Data
    # ---------------------
    data_module = fetch_data_module(config=config)

    # ---------------------
    # Logging and Checkpoints
    # ---------------------
    logger = CSVLogger(save_dir='./validation_logs')
    in_res_hw = tuple(config.model.backbone.in_res_hw)
    input_padder = InputPadderFromShape(desired_hw=in_res_hw)
    # ckpt_path = Path(config.checkpoint)

    # ---------------------
    # Model
    # ---------------------
    
    module = fetch_model_module(config=config)

    print(ModelSummary(module,max_depth=2))
    ckpt_path = config.checkpoint
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
        if config.model.freeze:
            for param in module.mdl.backbone.parameters():
                param.requires_grad = False

            for param in module.mdl.fpn.parameters():
                param.requires_grad = False

            module.mdl.backbone.eval()
            module.mdl.fpn.eval()
        ckpt_path = None
    else:
        raise Exception("No checkpoint used")


    paths = [PATH/ "train" /path for path in os.listdir(PATH/"train")]
    paths.extend([PATH / "val" / path for path in os.listdir(PATH/"val")])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device)
    module.eval()
    total_count = 0
    event_count = 0
    for path in paths:
        events_path = path / "events"
        for ev_name in os.listdir(events_path):
            event_count += 1
            event = np.load(events_path / ev_name)['arr_0']
            event = torch.from_numpy(event)
            event = event.unsqueeze(0)
            ev_tensors = input_padder.pad_tensor_ev_repr(event)
            ev_tensors = ev_tensors.to(device)
            preds, _, _ = module.mdl.backbone(ev_tensors)
            preds = module.mdl.fpn(preds)
            sparsity_mask = get_sparsity_mask(preds[0])
            sparsity_mask = sparsity_mask > 0.12
            print(f"{sparsity_mask.sum().item()=}")
            print(f"{sparsity_mask.shape=}")
            total_count += sparsity_mask.sum().item()
            # print(total_count)

    print(f"{total_count=}")
    print(f"{event_count=}")
    print(f"{total_count / event_count=}")


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
