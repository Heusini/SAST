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

class AugmentedDataset(Dataset):
    def __init__(
            self,
            original_dataset: Dataset,
            resolution_hw: Tuple[int, int],
            augmentation_config,
            ):
        self.dataset = original_dataset
        self.spatial_augmentor = RandomSpatialAugmentorGenX(
            dataset_hw=resolution_hw,
            automatic_randomization=True,
            augm_config=augmentation_config.random)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        return self.spatial_augmentor(item)
    
    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def build(dataset_config: DictConfig, dataset: Dataset):
        augmented_dataset = AugmentedDataset(
            dataset,
            tuple(dataset_config.resolution_hw),
            dataset_config.data_augmentation,
        )
        return augmented_dataset

