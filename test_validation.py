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
from pytorch_lightning.callbacks import ModelSummary
from typing import List, Tuple

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_data_module, fetch_model_module
from data.utils.types import DataType, LstmStates, ObjDetOutput, DatasetSamplingMode, BackboneFeatures
from utils.padding import InputPadderFromShape
from modules.utils.detection import BackboneFeatureSelector, EventReprSelector, RNNStates, Mode, mode_2_string, \
    merge_mixed_batches

from utils.timers import CudaTimer
from utils.timers import print_timing_info

import matplotlib.pyplot as plt

def extract_bounding_boxes(labels: np.ndarray) -> np.ndarray:                 
    # print(f"{labels=}")
    stacked = np.column_stack([labels[field] for field in labels.dtype.names])
    new_bbs = stacked[:, 1:5]                                                 
    if new_bbs.ndim == 1:                                                     
        new_bbs = np.expand_dims(new_bbs, axis=0)                             
    new_bbs[:, 2:] += new_bbs[:, :2]                                          
    # new_bbs = new_bbs.astype(np.int32)                                      
    return new_bbs

class Visualizer:
    def __init__(self, event_frame):
        self.gt_boxes = []
        self.dt_boxes = []
        self.event_frame = event_frame


    def add_gt_boxes(self, boxes):
        self.gt_boxes.extend(boxes.astype(np.int32).tolist())
        # print(self.gt_boxes)

    def add_dt_boxes(self, boxes):
        self.dt_boxes.extend(boxes.astype(np.int32).tolist())

    def draw(self):
        img = self.event_frame
        if len(img.shape) == 4:
            img = np.sum(img, axis=0)
        img = np.sum(img, axis=0)
        img = img / np.max(img)
        img = img * 255
        img = np.array(img, np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

        red = (0, 0, 255)
        green = (0, 255, 0)

        if len(self.gt_boxes) > 0:
            img = bbv.draw_multiple_rectangles(img, self.gt_boxes, bbox_color=green, thickness=1)
        if len(self.dt_boxes) > 0:
            img = bbv.draw_multiple_rectangles(img, self.dt_boxes, bbox_color=red, thickness=1)

        return img

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
    module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config}, strict=True)
    module.setup('validate')
    module.eval()
    # Get a batch (or a single sample wrapped as batch)
    data_module.setup('validate')
    val_loader = data_module.val_dataloader()

    in_res_hw = tuple(config.model.backbone.in_res_hw)
    input_padder = InputPadderFromShape(desired_hw=in_res_hw)
    # batch = next(iter(val_loader))  # or your own custom input
    # No gradient computation needed during inference
    count = 0
    skip = 50
    samples = 0
    gpu = torch.device(f"cuda:{gpus[0]}")
    for batch in val_loader:
        if count > skip:
            with CudaTimer(gpu, "Total"):
                output = module._val_test_step_impl(batch, Mode.VAL)
            visualizer = Visualizer(output[ObjDetOutput.EV_REPR].numpy())
            # print(output[ObjDetOutput.LABELS_PROPH]['t'])

            gt_boxes = extract_bounding_boxes(output[ObjDetOutput.LABELS_PROPH])
            dt_boxes = extract_bounding_boxes(output[ObjDetOutput.PRED_PROPH])
            visualizer.add_gt_boxes(gt_boxes)
            visualizer.add_dt_boxes(dt_boxes)
            img = visualizer.draw()
            cv2.imshow("window", img)
            if cv2.waitKey(1000) == ord("q"):
                cv2.destroyAllWindows()
                sys.exit(0)

        if count > skip+samples:
            break
        count += 1
    module.run_psee_evaluator(Mode.VAL)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
 
