from typing import Dict, Optional, Tuple, Union
import sys

import torch as th
import torch.nn as nn
from omegaconf import DictConfig

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from ..recurrent_backbone import build_recurrent_backbone
from ..yolox_extension.models.build import build_yolox_fpn, build_yolox_head
from utils.timers import TimerDummy as CudaTimer

from data.utils.types import BackboneFeatures, LstmStates
from .lwdetr import build as build_lwdetr
from util.box_ops import box_xyxy_to_cxcywh

class LWDETRDetector(th.nn.Module):
    def __init__(self,
                 model_cfg: DictConfig):
        super().__init__()
        backbone_cfg = model_cfg.backbone
        fpn_cfg = model_cfg.fpn
        head_cfg = model_cfg.head

        self.backbone = build_recurrent_backbone(backbone_cfg)

        # self.simple_rgbencode = SimpleRGBEncoder(64)
        # self.cross_attention = CrossAttention(64, 8)

        in_channels = self.backbone.get_stage_dims(fpn_cfg.in_stages)
        self.fpn = build_yolox_fpn(fpn_cfg, in_channels=in_channels)

        self.max_pool = nn.MaxPool2d(2, 2)

        strides = self.backbone.get_strides(fpn_cfg.in_stages)
        self.lwdetr, self.criterion, self.postprocessors = build_lwdetr(head_cfg)

    def forward_backbone(self,
                         x: th.Tensor,
                         previous_states: Optional[LstmStates] = None,
                         token_mask: Optional[th.Tensor] = None) -> \
            Tuple[BackboneFeatures, LstmStates, th.Tensor]:
        with CudaTimer(device=x.device, timer_name="Backbone"):
            backbone_features, states, p = self.backbone(x, previous_states, token_mask)
        return backbone_features, states, p

    def forward_fpn(self, backbone_features):
        device = next(iter(backbone_features.values())).device

        with CudaTimer(device=device, timer_name="FPN"):
            fpn_features = self.fpn(backbone_features)
        # print(f"{len(fpn_features)=}")
        # print(f"{fpn_features[0].shape=}")
        return fpn_features

    def get_sparsity_mask(self, fpn_layer: th.Tensor, threshold = 0.7):
        tokens = fpn_layer
        tokens = th.norm(tokens, dim=1)
        min_val = tokens.amin(dim=(-2, -1), keepdim=True)
        max_val = tokens.amax(dim=(-2, -1), keepdim=True)

        tokens = (tokens - min_val) / (max_val-min_val + 1e-8)
        if self.max_pool is not None:
            tokens = self.max_pool(tokens)
        #     print("max_pool")

        return tokens

    def forward_detect(self,
                       event_frame: th.Tensor,
                       rgb_image: th.Tensor,
                       sparsity_mask,
                       targets: Optional[th.Tensor] = None) -> \
            Tuple[th.Tensor, Union[Dict[str, th.Tensor], None]]:

        device = next(iter(event_frame)).device
        dtype = next(self.parameters()).dtype
        # print(f"{len(event_frame)=}")
        # print(f"{event_frame[0].shape=}")
        event_frame = th.vstack(event_frame).to(dtype)
        # print(f"{event_frame.shape=}")
        with CudaTimer(device=device, timer_name="HEAD + Loss"):
            outputs = self.lwdetr(event_frame, sparsity_mask, targets)

        loss_dict = self.criterion(outputs, targets)
        predictions = self.postprocessors['bbox'](outputs)
        weight_dict = self.criterion.weight_dict

        loss = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        loss_dict['loss'] = loss

        return predictions, loss_dict

    def forward(self,
                x: th.Tensor,
                previous_states: Optional[LstmStates] = None,
                retrieve_detections: bool = True,
                targets: Optional[th.Tensor] = None) -> \
            Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None], LstmStates, th.Tensor]:
        backbone_features, states, p, _, _ = self.forward_backbone(x, previous_states)
        outputs, losses = None, None
        if not retrieve_detections:
            assert targets is None
            return outputs, losses, states
        outputs, losses = self.forward_detect(backbone_features=backbone_features, targets=targets)
        return outputs, losses, states, p

