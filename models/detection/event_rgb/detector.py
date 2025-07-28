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

from ..rgb_yolo.yolo_pafpn import YOLOPAFPN

from data.utils.types import BackboneFeatures, LstmStates


class EventRGBDetector(th.nn.Module):
    def __init__(self,
                 model_cfg: DictConfig):
        super().__init__()
        backbone_cfg = model_cfg.backbone
        fpn_cfg = model_cfg.fpn
        head_cfg = model_cfg.head

        self.backbone = build_recurrent_backbone(backbone_cfg)

        in_channels = self.backbone.get_stage_dims(fpn_cfg.in_stages)
        strides = self.backbone.get_strides(fpn_cfg.in_stages)
        self.fpn = build_yolox_fpn(fpn_cfg, in_channels=in_channels)
        self.rgb_fpn = YOLOPAFPN(0.33, 0.5) 
        yolox_path = "./yolox_s.pth"
        # yolox_path = None
        if yolox_path:
            ckpt = th.load(yolox_path, map_location='cpu')
            state_dict = ckpt["model"]

            self.rgb_fpn.load_state_dict(state_dict, strict=False)
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
        with CudaTimer(device=device, timer_name="FPN"):
            fpn_features = self.fpn(backbone_features)

        # print(f"{rgb_image.norm(p=2)=}")
        # print(f"{rgb_image.shape=}")
        # for i in range(len(fpn_features)):
        #     print(f"{fpn_features[i].shape=}")
        #     print(f"{fpn_features[i].norm(p=2)=}")
        rgb_features = self.rgb_fpn(rgb_image)
        # for i in range(len(rgb_features)):
        #     print(f"{rgb_features[i].shape=}")
        #     print(f"{rgb_features[i].norm(p=2)=}")
        # sys.exit(0)

        features = []
        for f, r in zip(fpn_features, rgb_features):
            intermediate_features = th.add(f, r)
            features.append(intermediate_features)

        with CudaTimer(device=device, timer_name="HEAD + Loss"):
            outputs, losses = self.yolox_head(features, targets)
        return outputs, losses

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
