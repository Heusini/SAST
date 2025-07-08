from typing import Dict, Optional, Tuple, Union
import sys

import torch as th
from omegaconf import DictConfig

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from ..recurrent_backbone import build_recurrent_backbone
from ..yolox_extension.models.build import build_yolox_fpn, build_yolox_head
from utils.timers import TimerDummy as CudaTimer

from data.utils.types import BackboneFeatures, LstmStates
from .multiattention import CrossAttention, SimpleRGBEncoder, MLP, FPN


class EventRGBDetector(th.nn.Module):
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
        self.rgb_fpn = FPN(3)
        strides = self.backbone.get_strides(fpn_cfg.in_stages)
        in_channels = (128, 256, 512)
        self.yolox_head = build_yolox_head(head_cfg, in_channels=in_channels, strides=strides)

    def forward_backbone(self,
                         x: th.Tensor,
                         previous_states: Optional[LstmStates] = None,
                         token_mask: Optional[th.Tensor] = None) -> \
            Tuple[BackboneFeatures, LstmStates, th.Tensor]:
        with CudaTimer(device=x.device, timer_name="Backbone"):
            backbone_features, states, p = self.backbone(x, previous_states, token_mask)
        return backbone_features, states, p

    def forward_detect(self,
                       backbone_features: BackboneFeatures,
                       rgb_image: th.Tensor,
                       targets: Optional[th.Tensor] = None) -> \
            Tuple[th.Tensor, Union[Dict[str, th.Tensor], None]]:
        device = next(iter(backbone_features.values())).device
        # embedded_img = self.simple_rgbencode(rgb_image)
        # print(f"{embedded_img.shape=}")
        
        # for k in backbone_features.keys():
        #     print(f"{backbone_features[k].shape=}")
        # event_data = th.norm(backbone_features[1], dim=1)
        # event_data = backbone_features[1]
        # print(f"{event_data.shape=}")

        # self.cross_attention(event_data, embedded_img, embedded_img)

        # new_features = dict()
        stages = self.fpn.in_features
        # for k in stages[1:]:
        #     new_features[k] = backbone_features[k]

        # new_features[0] = rgb_image
        # print(f"{rgb_image.shape=}")
        # print(f"{new_features[1].shape=}")
        # print(f"{new_features[2].shape=}")
        # print(f"{new_features[3].shape=}")

        with CudaTimer(device=device, timer_name="FPN"):
            fpn_features = self.fpn(backbone_features)
        # if self.training:
        # assert targets is not None
        # for k in range(len(stages)):
        #     print(f"{fpn_features[k].shape=}")

        rgb_features = self.rgb_fpn(rgb_image)
        # for k in range(len(rgb_features)):
        #     print(f"{rgb_features[k].shape=}")

        fused_feats = [th.cat([a, b], dim=1) for a, b in zip(fpn_features, rgb_features)]
        # for k in range(len(fused_feats)):
        #     print(f"{fused_feats[k].shape=}")


        with CudaTimer(device=device, timer_name="HEAD + Loss"):
            outputs, losses = self.yolox_head(fused_feats, targets)
        return outputs, losses

        # with CudaTimer(device=device, timer_name="HEAD"):
        #     outputs, losses = self.yolox_head(fpn_features)
        # assert losses is None
        # return outputs, losses

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
