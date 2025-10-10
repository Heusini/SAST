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
                 objframe_idx_2_label_idx: th.Tensor,
                 input_size_hw: Tuple[int, int],
                 downsample_factor: Optional[float] = None):
        super().__init__(object_labels=object_labels, input_size_hw=input_size_hw)
        assert objframe_idx_2_label_idx.dtype == th.int64
        assert objframe_idx_2_label_idx.dim() == 1

        self.objframe_idx_2_label_idx = objframe_idx_2_label_idx
        self.downsample_factor = downsample_factor
        if self.downsample_factor is not None:
            assert self.downsample_factor > 1
        self.clamp_to_frame_()

    @staticmethod
    def from_structured_array(object_labels: np.ndarray,
                              objframe_idx_2_label_idx: np.ndarray,
                              input_size_hw: Tuple[int, int],
                              downsample_factor: Optional[float] = None) -> ObjectLabelFactory:
        np_labels = [object_labels[key].astype('float32') for key in ObjectLabels._str2idx.keys()]
        np_labels = rearrange(np_labels, 'fields L -> L fields')
        torch_labels = th.from_numpy(np_labels)
        objframe_idx_2_label_idx = th.from_numpy(objframe_idx_2_label_idx.astype('int64'))
        assert objframe_idx_2_label_idx.numel() == np.unique(object_labels['t']).size
        return ObjectLabelFactory(object_labels=torch_labels,
                                  objframe_idx_2_label_idx=objframe_idx_2_label_idx,
                                  input_size_hw=input_size_hw,
                                  downsample_factor=downsample_factor)

    def __len__(self):
        return len(self.objframe_idx_2_label_idx)

    def __getitem__(self, item: int) -> ObjectLabels:
        assert item >= 0
        length = len(self)
        assert length > 0
        assert item < length
        is_last_item = (item == length - 1)

        from_idx = self.objframe_idx_2_label_idx[item]
        to_idx = self.object_labels.shape[0] if is_last_item else self.objframe_idx_2_label_idx[item + 1]
        assert to_idx > from_idx
        object_labels = ObjectLabels(
            object_labels=self.object_labels[from_idx:to_idx].clone(),
            input_size_hw=self.input_size_hw)
        if self.downsample_factor is not None:
            object_labels.scale_(scaling_multiplier=1 / self.downsample_factor)
        return object_labels
