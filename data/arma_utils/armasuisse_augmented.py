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

class ArmasuisseAugmented(Dataset):
    def __init__(
            self,
            path: Path,
            sequence_length: int,
            resolution_hw: Tuple[int, int],
            augmentation_config,
            ):
        self.armasuisse_dataset = ArmasuisseDataset(path, sequence_length, resolution_hw)
        self.spatial_augmentor = RandomSpatialAugmentorGenX(
            dataset_hw=resolution_hw,
            automatic_randomization=True,
            augm_config=augmentation_config.random)




    def __getitem__(self, index: int):
        item = self.armasuisse_dataset[index]
        return self.spatial_augmentor(item)
    
    def __len__(self) -> int:
        return len(self.armasuisse_dataset)

    @staticmethod
    def build(dataset_mode: DatasetMode, dataset_config: DictConfig):
        path = Path(dataset_config.path)
        assert path.exists(), f"provided Armasuisse path {path} does not exist"
        PATHS = {
            "train": path / "train",
            "val": path / "val"
        }
        mode2str = {DatasetMode.TRAIN: 'train',
                    DatasetMode.VALIDATION: 'val',
                    DatasetMode.TESTING: 'test'}

        data_folder = PATHS[mode2str[dataset_mode]]
        assert data_folder.is_dir(), f"Train folder ({data_folder}) doesn't exist maybe structure is wrong of the preprocessed data"
        dataset = ArmasuisseAugmented(
            data_folder,
            dataset_config.sequence_length,
            tuple(dataset_config.resolution_hw),
            dataset_config.data_augmentation,
        )
        return dataset

