from __future__ import annotations

from typing import List, Tuple, Union, Optional

import math
import numpy as np
import torch as th
from einops import rearrange
from torch.nn.functional import pad


class ObjectLabelBase:
    _str2idx = {
        't': 0,
        'x': 1,
        'y': 2,
        'w': 3,
        'h': 4,
        'class_id': 5,
        'class_confidence': 6,
    }

    def __init__(self,
                 object_labels: th.Tensor,
                 input_size_hw: Tuple[int, int]):
        assert isinstance(object_labels, th.Tensor)
        assert object_labels.dtype in {th.float32, th.float64}
        assert object_labels.ndim == 2
        assert object_labels.shape[-1] == len(self._str2idx)
        assert isinstance(input_size_hw, tuple)
        assert len(input_size_hw) == 2

        self.object_labels = object_labels
        self._input_size_hw = input_size_hw
        self._is_numpy = False

    def clamp_to_frame_(self):
        ht, wd = self.input_size_hw
        x0 = th.clamp(self.x, min=0, max=wd - 1)
        y0 = th.clamp(self.y, min=0, max=ht - 1)
        x1 = th.clamp(self.x + self.w, min=0, max=wd - 1)
        y1 = th.clamp(self.y + self.h, min=0, max=ht - 1)
        w = x1 - x0
        h = y1 - y0
        assert th.all(w > 0)
        assert th.all(h > 0)
        self.x = x0
        self.y = y0
        self.w = w
        self.h = h

    def remove_flat_labels_(self):
        keep = (self.w > 0) & (self.h > 0)
        self.object_labels = self.object_labels[keep]

    @classmethod
    def create_empty(cls):
        # This is useful to represent cases where no labels are available.
        return ObjectLabelBase(object_labels=th.empty((0, len(cls._str2idx))), input_size_hw=(0, 0))

    def _assert_not_numpy(self):
        assert not self._is_numpy, "Labels have been converted numpy. \
        Numpy is not supported for the intended operations."

    def to(self, *args, **kwargs):
        # This function executes torch.to on self tensors and returns self.
        self._assert_not_numpy()
        # This will be used by Pytorch Lightning to transfer to the relevant device
        self.object_labels = self.object_labels.to(*args, **kwargs)
        return self

    def numpy_(self) -> None:
        """
        In place conversion to numpy (detach + to cpu + to numpy).
        Cannot be undone.
        """
        self._is_numpy = True
        self.object_labels = self.object_labels.detach().cpu().numpy()

    @property
    def input_size_hw(self) -> Tuple[int, int]:
        return self._input_size_hw

    @input_size_hw.setter
    def input_size_hw(self, height_width: Tuple[int, int]):
        assert isinstance(height_width, tuple)
        assert len(height_width) == 2
        assert height_width[0] > 0
        assert height_width[1] > 0
        self._input_size_hw = height_width

    def get(self, request: str):
        assert request in self._str2idx
        return self.object_labels[:, self._str2idx[request]]

    @property
    def t(self):
        return self.object_labels[:, self._str2idx['t']]

    @property
    def x(self):
        return self.object_labels[:, self._str2idx['x']]

    @x.setter
    def x(self, value: Union[th.Tensor, np.ndarray]):
        self.object_labels[:, self._str2idx['x']] = value

    @property
    def y(self):
        return self.object_labels[:, self._str2idx['y']]

    @y.setter
    def y(self, value: Union[th.Tensor, np.ndarray]):
        self.object_labels[:, self._str2idx['y']] = value

    @property
    def w(self):
        return self.object_labels[:, self._str2idx['w']]

    @w.setter
    def w(self, value: Union[th.Tensor, np.ndarray]):
        self.object_labels[:, self._str2idx['w']] = value

    @property
    def h(self):
        return self.object_labels[:, self._str2idx['h']]

    @h.setter
    def h(self, value: Union[th.Tensor, np.ndarray]):
        self.object_labels[:, self._str2idx['h']] = value

    @property
    def class_id(self):
        return self.object_labels[:, self._str2idx['class_id']]

    @property
    def class_confidence(self):
        return self.object_labels[:, self._str2idx['class_confidence']]

    @property
    def dtype(self):
        return self.object_labels.dtype

    @property
    def device(self):
        return self.object_labels.device
