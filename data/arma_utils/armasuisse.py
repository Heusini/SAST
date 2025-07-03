# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Face dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
import os
import sys
from pathlib import Path
from typing import Any, List, Tuple

import torch
# import torch.utils.data
from torch.utils.data import ConcatDataset, Dataset
from omegaconf import DictConfig
import torchvision
import numpy as np
import cv2
from data.utils.types import DataType, LoaderDataDictGenX, DatasetMode
from data.arma_utils.labels import ObjectLabelFactory
from data.utils.object_labels import ObjectLabels
from data.utils.sparsely_batched_object_labels import SparselyBatchedObjectLabels

class Sequence:
    def __init__(self, data_paths: List[Path], label_paths: List[Path]):
        assert len(data_paths) == len(label_paths)
        self.data_paths = data_paths
        self.label_paths = label_paths
        self.size = len(data_paths)
    def __len__(self):
        return self.size

# this is specific to the armasuisse preprocessed data
def create_sequences(path: Path, sequence_length: int):
    seq_list = list()
    event_folder = Path("events")
    label_folder = Path("labels")
    for dir in os.listdir(path):
        event_path = path / dir / event_folder
        label_path = path / dir / label_folder

        event_files = os.listdir(event_path)
        label_files = os.listdir(label_path)

        # sorts by length first and than alphanumerical
        event_files.sort(key=lambda item: (len(item), item))
        label_files.sort(key=lambda item: (len(item), item))
        assert_msg = f"event_len({len(event_files)}) != label_len({label_files}) for\n {event_path} and\n {label_path}"
        assert len(event_files) > 0
        assert len(event_files) == len(label_files), assert_msg


        # we start at sequence_length to not run out of elements at the end of the files
        for index in range(sequence_length, len(event_files), sequence_length):
            start = index - sequence_length
            event_list = [event_path / event_file for event_file in event_files[start:start+sequence_length]]
            label_list = [label_path / label_file for label_file in label_files[start:start+sequence_length]]
            sequence = Sequence(event_list, label_list)
            seq_list.append(sequence)

    return seq_list

class ArmasuisseDataset(Dataset):
    def __init__(
            self, 
            path: Path,
            sequence_length: int,
            resolution_hw: Tuple[int, int],
            ) -> None:
        assert path.is_dir()
        self.path = path
        self.sequence_length = sequence_length
        self.sequences = create_sequences(path, sequence_length)
        self.resolution_hw = resolution_hw

    def __getitem__(self, index: int) -> LoaderDataDictGenX:
        sequence = self.sequences[index]

        events = list()
        labels = list()
        # event_paths = list()
        # label_paths = list()
        for event_path in sequence.data_paths:
            event = np.load(event_path)
            # event_paths.append(str(event_path))
            event = event[list(event.keys())[0]]
            event = torch.from_numpy(event)
            events.append(event)

        for label_path in sequence.label_paths:
            label = np.load(label_path)
            # label_paths.append(str(label_path))
            label = label[list(label.keys())[0]]
            label = ObjectLabelFactory.from_structured_array(label,
                                                             self.resolution_hw,
                                                             None)
            labels.append(label.get_object_labels())


        sparse_labels = SparselyBatchedObjectLabels(labels)
        is_first_sample = True
        is_padded_mask = [False] * len(events)

        out = {
            DataType.EV_REPR: events,
            DataType.OBJLABELS_SEQ: sparse_labels,
            DataType.IS_FIRST_SAMPLE: is_first_sample,
            DataType.IS_PADDED_MASK: is_padded_mask,
            # DataType.EVENT_PATH: event_paths,
            # DataType.LABEL_PATH: label_paths,
        }
        return out

    def __len__(self) -> int:
        return len(self.sequences)

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
        dataset = ArmasuisseDataset(
            data_folder,
            dataset_config.sequence_length,
            tuple(dataset_config.resolution_hw),
        )
        return dataset
