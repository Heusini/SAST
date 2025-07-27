from __future__ import annotations

from typing import List, Tuple, Union, Optional

import math
import numpy as np
import torch as th
from einops import rearrange
from torch.nn.functional import pad
from data.utils.object_label_base import ObjectLabelBase

class ObjectLabels(ObjectLabelBase):
    def __init__(self,
                 object_labels: th.Tensor,
                 input_size_hw: Tuple[int, int]):
        super().__init__(object_labels=object_labels, input_size_hw=input_size_hw)

    def __len__(self) -> int:
        return self.object_labels.shape[0]

    def rotate_(self, angle_deg: float):
        if len(self) == 0:
            return self
        # (x0,y0)---(x1,y0)   p00---p10
        #  |             |    |       |
        #  |             |    |       |
        # (x0,y1)---(x1,y1)   p01---p11
        p00 = th.stack((self.x, self.y), dim=1)
        p10 = th.stack((self.x + self.w, self.y), dim=1)
        p01 = th.stack((self.x, self.y + self.h), dim=1)
        p11 = th.stack((self.x + self.w, self.y + self.h), dim=1)
        # points: 4 x N x 2
        points = th.stack((p00, p10, p01, p11), dim=0)

        cx = self._input_size_hw[1] // 2
        cy = self._input_size_hw[0] // 2
        center = th.tensor([cx, cy], device=self.device)

        angle_rad = angle_deg / 180 * math.pi
        # counter-clockwise rotation
        rot_matrix = th.tensor([[math.cos(angle_rad), math.sin(angle_rad)],
                                [-math.sin(angle_rad), math.cos(angle_rad)]], device=self.device)

        points = points - center
        points = th.einsum('ij,pnj->pni', rot_matrix, points)
        points = points + center

        height, width = self.input_size_hw
        x0 = th.clamp(th.min(points[..., 0], dim=0)[0], min=0, max=width - 1)
        y0 = th.clamp(th.min(points[..., 1], dim=0)[0], min=0, max=height - 1)
        x1 = th.clamp(th.max(points[..., 0], dim=0)[0], min=0, max=width - 1)
        y1 = th.clamp(th.max(points[..., 1], dim=0)[0], min=0, max=height - 1)

        self.x = x0
        self.y = y0
        self.w = x1 - x0
        self.h = y1 - y0

        self.remove_flat_labels_()

        assert th.all(self.x >= 0)
        assert th.all(self.y >= 0)
        assert th.all(self.x + self.w <= self.input_size_hw[1] - 1)
        assert th.all(self.y + self.h <= self.input_size_hw[0] - 1)

    def zoom_in_and_rescale_(self, zoom_coordinates_x0y0: Tuple[int, int], zoom_in_factor: float):
        """
        1) Computes a new smaller canvas size: original canvas scaled by a factor of 1/zoom_in_factor (downscaling)
        2) Places the smaller canvas inside the original canvas at the top-left coordinates zoom_coordinates_x0y0
        3) Extract the smaller canvas and rescale it back to the original resolution
        """
        if len(self) == 0:
            return self
        assert len(zoom_coordinates_x0y0) == 2
        assert zoom_in_factor >= 1
        if zoom_in_factor == 1:
            return self
        z_x0, z_y0 = zoom_coordinates_x0y0
        h_orig, w_orig = self.input_size_hw
        assert 0 <= z_x0 <= w_orig - 1
        assert 0 <= z_y0 <= h_orig - 1
        zoom_window_h, zoom_window_w = tuple(x / zoom_in_factor for x in self.input_size_hw)
        z_x1 = min(z_x0 + zoom_window_w, w_orig - 1)
        assert z_x1 <= w_orig - 1, f'{z_x1=} is larger than {w_orig-1=}'
        z_y1 = min(z_y0 + zoom_window_h, h_orig - 1)
        assert z_y1 <= h_orig - 1, f'{z_y1=} is larger than {h_orig-1=}'

        x0 = th.clamp(self.x, min=z_x0, max=z_x1 - 1)
        y0 = th.clamp(self.y, min=z_y0, max=z_y1 - 1)

        x1 = th.clamp(self.x + self.w, min=z_x0, max=z_x1 - 1)
        y1 = th.clamp(self.y + self.h, min=z_y0, max=z_y1 - 1)

        self.x = x0 - z_x0
        self.y = y0 - z_y0
        self.w = x1 - x0
        self.h = y1 - y0
        self.input_size_hw = (zoom_window_h, zoom_window_w)

        self.remove_flat_labels_()

        self.scale_(scaling_multiplier=zoom_in_factor)

    def zoom_out_and_rescale_(self, zoom_coordinates_x0y0: Tuple[int, int], zoom_out_factor: float):
        """
        1) Scales the input by a factor of 1/zoom_out_factor (i.e. reduces the canvas size)
        2) Places the downscaled canvas into the original canvas at the top-left coordinates zoom_coordinates_x0y0
        """
        if len(self) == 0:
            return self
        assert len(zoom_coordinates_x0y0) == 2
        assert zoom_out_factor >= 1
        if zoom_out_factor == 1:
            return self

        h_orig, w_orig = self.input_size_hw
        self.scale_(scaling_multiplier=1 / zoom_out_factor)

        self.input_size_hw = (h_orig, w_orig)
        z_x0, z_y0 = zoom_coordinates_x0y0
        assert 0 <= z_x0 <= w_orig - 1
        assert 0 <= z_y0 <= h_orig - 1

        self.x = self.x + z_x0
        self.y = self.y + z_y0

    def scale_(self, scaling_multiplier: float):
        if len(self) == 0:
            return self
        assert scaling_multiplier > 0
        if scaling_multiplier == 1:
            return self
        img_ht, img_wd = self.input_size_hw
        new_img_ht = scaling_multiplier * img_ht
        new_img_wd = scaling_multiplier * img_wd
        self.input_size_hw = (new_img_ht, new_img_wd)
        x1 = th.clamp((self.x + self.w) * scaling_multiplier, max=new_img_wd - 1)
        y1 = th.clamp((self.y + self.h) * scaling_multiplier, max=new_img_ht - 1)
        self.x = self.x * scaling_multiplier
        self.y = self.y * scaling_multiplier

        self.w = x1 - self.x
        self.h = y1 - self.y

        self.remove_flat_labels_()

    def flip_lr_(self) -> None:
        if len(self) == 0:
            return self
        self.x = self.input_size_hw[1] - 1 - self.x - self.w

    def get_labels_as_tensors(self, format_: str = 'yolox') -> th.Tensor:
        self._assert_not_numpy()

        if format_ == 'yolox':
            out = th.zeros((len(self), 5), dtype=th.float32, device=self.device)
            if len(self) == 0:
                return out
            out[:, 0] = self.class_id
            out[:, 1] = self.x + 0.5 * self.w
            out[:, 2] = self.y + 0.5 * self.h
            out[:, 3] = self.w
            out[:, 4] = self.h
            return out
        else:
            raise NotImplementedError

    def get_labels_xyxy(self) -> th.Tensor:
        out = th.zeros((len(self), 5), device=self.device)
        if len(self) == 0:
            return out
        out[:, 0] = self.x
        out[:, 1] = self.y
        out[:, 2] = self.x + self.w
        out[:, 3] = self.y + self.h
        out[:, 4] = self.class_id
        return out

    @staticmethod
    def get_labels_as_batched_tensor(obj_label_list: List[ObjectLabels], format_: str = 'yolox') -> th.Tensor:
        num_object_frames = len(obj_label_list)
        if num_object_frames == 0:
            print("lol1")
            return None
        # assert num_object_frames > 0
        max_num_labels_per_object_frame = max([len(x) for x in obj_label_list])
        # assert max_num_labels_per_object_frame > 0

        if format_ == 'yolox':
            tensor_labels = []
            for labels in obj_label_list:
                obj_labels_tensor = labels.get_labels_as_tensors(format_=format_)
                num_to_pad = max_num_labels_per_object_frame - len(labels)
                padded_labels = pad(obj_labels_tensor, (0, 0, 0, num_to_pad), mode='constant', value=0)
                tensor_labels.append(padded_labels)
            tensor_labels = th.stack(tensors=tensor_labels, dim=0)
            return tensor_labels
        else:
            raise NotImplementedError
