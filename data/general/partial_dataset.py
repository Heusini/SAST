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

from omegaconf import DictConfig
from data.utils.types import DataType, LoaderDataDictGenX, DatasetMode
from data.utils.augmentor import RandomSpatialAugmentorGenX
from data.arma_utils.labels import ObjectLabelFactory
from data.utils.sparsely_batched_object_labels import SparselyBatchedObjectLabels
from data.arma_utils.armasuisse import ArmasuisseDataset
from numpy.random import default_rng


class PartialDataset(Dataset):
    def __init__(
            self,
            original_dataset: Dataset,
            percentage: float,
            randomize: bool = False,
            seed: int = 42,
            ):
        self.dataset = original_dataset
        self.percentage = percentage
        self.random = randomize
        rng = default_rng(seed)
        self.len = np.ceil(len(original_dataset) * percentage).astype(np.int32)
        self.numbers = rng.choice(len(original_dataset), size=self.len, replace=False)


    def __getitem__(self, index: int):
        if self.random:
            index = self.numbers[index]
        return self.dataset[index]

    def __len__(self) -> int:
        return self.len
