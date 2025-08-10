from typing import Dict, Optional, Tuple, Union
import sys

import torch as th
from omegaconf import DictConfig
from utils.padding import InputPadderFromShape

try:
    from torch import compile as th_compile
except ImportError:
    th_compile = None

from ..recurrent_backbone import build_recurrent_backbone
from ..yolox_extension.models.build import build_yolox_fpn, build_yolox_head
from utils.timers import CudaTimer

from ..rgb_yolo.yolo_pafpn import YOLOPAFPN

from data.utils.types import BackboneFeatures, LstmStates


class RGBDetector(th.nn.Module):
    def __init__(self,
                 model_cfg: DictConfig):
        super().__init__()
        backbone_cfg = model_cfg.backbone
        fpn_cfg = model_cfg.fpn
        head_cfg = model_cfg.head

        backbone = build_recurrent_backbone(backbone_cfg)
        self.backbone = YOLOPAFPN(0.33, 0.5) 

        in_channels = backbone.get_stage_dims(fpn_cfg.in_stages)
        strides = backbone.get_strides(fpn_cfg.in_stages)

        yolox_path = "./yolox_s.pth"
        # yolox_path = None
        if yolox_path:
            ckpt = th.load(yolox_path, map_location='cpu')
            state_dict = ckpt["model"]
            backbone_str = "backbone."
            backbone_dict = {k.replace(backbone_str, "", 1): v 
                         for k, v in state_dict.items()}

            self.backbone.load_state_dict(backbone_dict, strict=False)
        self.yolox_head = build_yolox_head(head_cfg, in_channels=in_channels, strides=strides)

    def forward_detect(self,
                       rgb_image: th.Tensor,
                       targets: Optional[th.Tensor] = None) -> \
            Tuple[th.Tensor, Union[Dict[str, th.Tensor], None]]:
        device = rgb_image.device
        rgb_features = self.backbone(rgb_image)

        outputs, losses = self.yolox_head(rgb_features, targets)
        return outputs, losses

    def forward(self,
                x: th.Tensor,
                rgb_image: th.Tensor,
                previous_states: Optional[LstmStates] = None,
                retrieve_detections: bool = True,
                targets: Optional[th.Tensor] = None) -> \
            Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None], LstmStates, th.Tensor]:
        with CudaTimer(th.device('cuda'), "PAFPN"):
            rgb_image, _ = InputPadderFromShape._pad_tensor_impl(rgb_image, (384, 640), mode='constant', value=0)
            rgb_features = self.backbone(rgb_image)
        with CudaTimer(th.device('cuda'), "YOLOX"):
            predictions, _ = self.yolox_head(rgb_features, None)

        return predictions


