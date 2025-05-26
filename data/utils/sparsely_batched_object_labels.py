from __future__ import annotations

from typing import List, Tuple, Union, Optional

import math
import numpy as np
import torch as th
from einops import rearrange
from torch.nn.functional import pad

class SparselyBatchedObjectLabels:
    def __init__(self, sparse_object_labels_batch: List[Optional[ObjectLabels]]):
        # Can contain None elements that indicate missing labels.
        for entry in sparse_object_labels_batch:
            assert isinstance(entry, ObjectLabels) or entry is None
        self.sparse_object_labels_batch = sparse_object_labels_batch
        self.set_empty_labels_to_none_()

    def __len__(self) -> int:
        return len(self.sparse_object_labels_batch)

    def __iter__(self):
        return iter(self.sparse_object_labels_batch)

    def __getitem__(self, item: int) -> Optional[ObjectLabels]:
        if item < 0 or item >= len(self):
            raise IndexError(f'Index ({item}) out of range (0, {len(self) - 1})')
        return self.sparse_object_labels_batch[item]

    def __add__(self, other: SparselyBatchedObjectLabels):
        sparse_object_labels_batch = self.sparse_object_labels_batch + other.sparse_object_labels_batch
        return SparselyBatchedObjectLabels(sparse_object_labels_batch=sparse_object_labels_batch)

    def set_empty_labels_to_none_(self):
        for idx, obj_label in enumerate(self.sparse_object_labels_batch):
            if obj_label is not None and len(obj_label) == 0:
                self.sparse_object_labels_batch[idx] = None

    @property
    def input_size_hw(self) -> Optional[Union[Tuple[int, int], Tuple[float, float]]]:
        for obj_labels in self.sparse_object_labels_batch:
            if obj_labels is not None:
                return obj_labels.input_size_hw
        return None

    def zoom_in_and_rescale_(self, *args, **kwargs):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].zoom_in_and_rescale_(*args, **kwargs)
        # We may have deleted labels. If no labels are left, set the object to None
        self.set_empty_labels_to_none_()

    def zoom_out_and_rescale_(self, *args, **kwargs):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].zoom_out_and_rescale_(*args, **kwargs)

    def rotate_(self, *args, **kwargs):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].rotate_(*args, **kwargs)

    def scale_(self, *args, **kwargs):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].scale_(*args, **kwargs)
        # We may have deleted labels. If no labels are left, set the object to None
        self.set_empty_labels_to_none_()

    def flip_lr_(self):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].flip_lr_()

    def to(self, *args, **kwargs):
        for idx, entry in enumerate(self.sparse_object_labels_batch):
            if entry is not None:
                self.sparse_object_labels_batch[idx].to(*args, **kwargs)
        return self

    def get_valid_labels_and_batch_indices(self) -> Tuple[List[ObjectLabels], List[int]]:
        out = list()
        valid_indices = list()
        for idx, label in enumerate(self.sparse_object_labels_batch):
            if label is not None:
                out.append(label)
                valid_indices.append(idx)
        return out, valid_indices

    @staticmethod
    def transpose_list(list_of_sparsely_batched_object_labels: List[SparselyBatchedObjectLabels]) \
            -> List[SparselyBatchedObjectLabels]:
        return [SparselyBatchedObjectLabels(list(labels_as_tuple)) for labels_as_tuple \
                in zip(*list_of_sparsely_batched_object_labels)]
