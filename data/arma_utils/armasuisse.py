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
import torchvision
import numpy as np
import cv2
from data.utils.types import DataType, LoaderDataDictGenX
from data.arma_utils.labels import ObjectLabelFactory, SparselyBatchedObjectLabels

import data.arma_utils.transforms as T
from data.arma_utils.box_ops import box_cxcywh_to_xyxy

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
    event_folder = Path("event_representation")
    label_folder = Path("labels")
    for dir in os.listdir(path):
        event_path = path / dir / event_folder
        label_path = path / dir / label_folder

        event_files = os.listdir(event_path)
        label_files = os.listdir(label_path)
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

class ArmasuisseRandom(Dataset):
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
        for event_path in sequence.data_paths:
            event = torch.load(event_path)
            events.append(event)

        for label_path in sequence.label_paths:
            label = np.load(label_path)
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
        }
        return out

    def __len__(self) -> int:
        return len(self.sequences)

class ConvertArmasuissePolysToMask(object):
    def __init__(self):
        pass

    def __call__(self, data, target):
        h = data.shape[1]
        w = data.shape[2]

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        # What does this parameter do
        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]

        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)


        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return data, target


def make_Armasuisse_transforms(data_set, new_size):
    # No normalisation for sparse processing
    means = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    stds = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    normalize = T.Compose([T.Normalize(means, stds)])

    if data_set == "train":
        return T.Compose(
            [
                # T.ResizeEvent(new_size[0], new_size[1]),
                T.RandomHorizontalFlipEvent(),
                T.RandomSelect(T.RandomZoomIn(), T.RandomZoomOut()),
                normalize,
            ]
        )

    if data_set == "val":
        return T.Compose(
            [
                T.ResizeEvent(new_size[0], new_size[1]),
                normalize,
            ]
        )

    raise ValueError(f"unknown {data_set}")


def build(dataset_mode, config):
    path = Path(config.path)
    assert path.exists(), f"provided Armasuisse path {path} does not exist"

    PATHS = {
        "train": path / "train",
        "val": path / "val"
    }

    data_folder = PATHS[dataset_mode]
    assert data_folder.is_dir(), f"Train folder ({data_folder}) doesn't exist maybe structure is wrong of the preprocessed data"
    new_size = max(config.resolution_hw)
    dataset = ArmasuisseRandom(
        data_folder,
        config.sequence_length,
        tuple(config.resolution_hw),
    )
    return dataset
