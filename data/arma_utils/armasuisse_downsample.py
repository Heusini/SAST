import os
import sys
from pathlib import Path
from typing import Any, List, Tuple

import torch
# import torch.utils.data
from torch.utils.data import ConcatDataset, Dataset
import torchvision
import numpy as np
import cv2
from data.utils.types import DataType, LoaderDataDictGenX
from data.arma_utils.labels import ObjectLabelFactory, SparselyBatchedObjectLabels

import data.arma_utils.transforms as T
from data.arma_utils.box_ops import box_cxcywh_to_xyxy





class ArmasuisseDownsample(Dataset):
    def __init__(
            self,
            path: Path,
            sequence_length: int,
            resolution_hw: Tuple[int, int],
            downsample_factor: float):

    self.armasuisse_dataset = ArmasuisseRandom(path, sequence_length, resolution_hw)




    def __getitem__(self, index: int):
        pass
        
    
    def __len__(self) -> int:
        return len(self.armasuisse_dataset)

