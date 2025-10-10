from __future__ import annotations

from typing import List, Tuple, Union, Optional

import math
import numpy as np
import torch as th
from einops import rearrange
from torch.nn.functional import pad

from data.utils.object_label_base import ObjectLabelBase
from data.utils.object_labels import ObjectLabels

class ObjectLabelFactory(ObjectLabelBase):
    def __init__(self,
                 object_labels: th.Tensor,
                 input_size_hw: Tuple[int, int],
                 downsample_factor: Optional[float] = None,
                 size = 0):
        super().__init__(object_labels=object_labels, input_size_hw=input_size_hw)

        self.downsample_factor = downsample_factor
        self.size = size
        if self.downsample_factor is not None:
            assert self.downsample_factor > 1
        self.clamp_to_frame_()

    @staticmethod
    def from_structured_array(object_labels: np.ndarray,
                              input_size_hw: Tuple[int, int],
                              downsample_factor: Optional[float] = None) -> ObjectLabelFactory:
        size = len(object_labels)
        np_labels = [object_labels[key].astype('float32') for key in ObjectLabels._str2idx.keys()]
        np_labels = rearrange(np_labels, 'fields L -> L fields')
        torch_labels = th.from_numpy(np_labels)
        return ObjectLabelFactory(object_labels=torch_labels,
                                  input_size_hw=input_size_hw,
                                  downsample_factor=downsample_factor,
                                  size=size)

    def __len__(self):
        return self.size

    def get_object_labels(self):
        if self.size == 0:
            object_labels = ObjectLabels(
                    object_labels = self.object_labels.clone(),
                    input_size_hw = self.input_size_hw)
            return object_labels
        else:
            object_labels = ObjectLabels(
                object_labels=self.object_labels.clone(),
                input_size_hw=self.input_size_hw)
            if self.downsample_factor is not None:
                object_labels.scale_(scaling_multiplier=1 / self.downsample_factor)
            return object_labels

    def __getitem__(self, item: int) -> ObjectLabels:
        if self.size == 0:
            object_labels = ObjectLabels(
                    object_labels = th.zeros(0,7),
                    input_size_hw = self.input_size_hw)
            return object_labels
        object_labels = ObjectLabels(
            object_labels=self.object_labels.clone(),
            input_size_hw=self.input_size_hw)
        if self.downsample_factor is not None:
            object_labels.scale_(scaling_multiplier=1 / self.downsample_factor)
        return object_labels
