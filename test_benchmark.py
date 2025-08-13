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
    ckpt_path = Path(config.checkpoint)

    # ---------------------
    # Model
    # ---------------------
    
    module = fetch_model_module(config=config)
    # module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config}, strict=True)
    print(ModelSummary(module,max_depth=2))

    in_res_hw = tuple(config.model.backbone.in_res_hw)
    input_padder = InputPadderFromShape(desired_hw=in_res_hw)

    data_module.setup('validate')
    val_loader = data_module.val_dataloader()
    data = next(iter(val_loader))['data']
    ev_tensor_sequence = data[DataType.EV_REPR]
    rgb_sequence = data[DataType.IMAGE]
    ev_tensors = ev_tensor_sequence[0]
    rgb_image = rgb_sequence[0]
    ev_tensors = input_padder.pad_tensor_ev_repr(ev_tensors)
    input_sample = ev_tensors
    print(input_sample.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_sample = input_sample.to(device)
    rgb_image = rgb_image.to(device)
    module.to(device)
    module.eval()
    print(module.device)
    print(input_sample.device)

    with torch.no_grad():
        output = measure_average_inference_time(module, input_sample, rgb_image, 500)
        print(output)



if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()

