import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path


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

@hydra.main(config_path='config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    # print('------ Configuration ------')
    # print(OmegaConf.to_yaml(config))
    # print('---------------------------')

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
    # ckpt_path = Path(config.checkpoint)

    # ---------------------
    # Model
    # ---------------------
    
    module = fetch_model_module(config=config)
    # module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config}, strict=True)
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
        pass
        # print("no checkpoint")

    in_res_hw = tuple(config.model.backbone.in_res_hw)
    input_padder = InputPadderFromShape(desired_hw=in_res_hw)


    input_event_path = "/datasets/sheusinger/st_stephan_360_640_20/train/2024_01_10_170509_himo_005/events/event_0.npz"
    input_img_path = "/datasets/sheusinger/st_stephan_360_640_20/train/2024_01_10_170509_himo_005/rgbs/rgb_0.npz"
    input_events = np.load(input_event_path)['arr_0']
    input_image = np.load(input_img_path)['arr_0']

    frame = torch.from_numpy(input_image)
    frame = frame.permute(-1, 0, 1)
    frame = frame / 255.0

    input_events = torch.from_numpy(input_events).unsqueeze(0)
    frame = frame.unsqueeze(0)

    ev_tensors = input_padder.pad_tensor_ev_repr(input_events)
    frame = input_padder.pad_tensor_ev_repr(frame) 
    input_sample = ev_tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_sample = input_sample.to(device)
    frame = frame.to(device)
    module.to(device)
    module.eval()

    if config.model.name == "lwdetr_official_rgb":
        tmp = frame
        frame = input_sample
        input_sample = tmp

    with torch.no_grad():
        output = measure_average_inference_time(module, input_sample, frame, 2000)
        print(output)



if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()

