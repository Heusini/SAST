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
from .lwdeter import build as build_lwdeter
from util.box_ops import box_xyxy_to_cxcywh

class LWDETERDetector(th.nn.Module):
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

        strides = self.backbone.get_strides(fpn_cfg.in_stages)
        self.lwdeter, self.criterion, self.postprocessors = build_lwdeter(head_cfg)

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

        return fpn_features

    # moved conversion to object_labels can maybe remove but validation of 
    # lwdetr format in object labels has to be done
    def lw_deter_labels(self, targets, im_h, im_w):
        # This function could be possibly done nicer
        # our dataloader outputs torch.empty((0,5)) for empty data
        # lw deter expects torch.empty(0) the target["labels"] if there is no label at all(in none of the batches)
        # if there are some empty lables (not all the empty labels and boxes are removed)

        new_targets = []
        h, w = im_h, im_w
        for image_id, gt in enumerate(targets):
            im_id = image_id + 1
            target = {}
            if targets.nelement() == 0:
                target["boxes"] = th.empty((0,4), device=gt.device)
                target["labels"] = th.empty(0,dtype=th.int64, device=gt.device)
                target["image_id"] = th.tensor(im_id, device=gt.device) 
                target["orig_size"] = th.as_tensor([int(h), int(w)])
                target["size"] = th.as_tensor([int(h), int(w)])
                new_targets.append(target)
            else:
                h,w = gt[0, 3:5]
                boxes = gt[:, 1:5]
                boxes[:, 2:] += boxes[:, :2]
                boxes = box_xyxy_to_cxcywh(boxes)
                boxes = boxes / th.tensor([im_w, im_h, im_w, im_h], device=gt.device)
                non_empty_mask = ~(boxes == 0).all(dim=1)
                boxes = boxes[non_empty_mask]
                labels = gt[:, 0].int()[non_empty_mask]
                target["boxes"] = boxes
                target["labels"] = labels
                target["image_id"] = th.tensor(im_id, device=gt.device) 
                target["orig_size"] = th.as_tensor([int(h), int(w)])
                target["size"] = th.as_tensor([int(h), int(w)])

                new_targets.append(target)
        return new_targets


    def forward_detect(self,
                       event_frame: th.Tensor,
                       rgb_image: th.Tensor,
                       sparsity_mask,
                       targets: Optional[th.Tensor] = None) -> \
            Tuple[th.Tensor, Union[Dict[str, th.Tensor], None]]:

        device = next(iter(event_frame)).device
        dtype = next(self.parameters()).dtype
        event_frame = th.vstack(event_frame).to(dtype)
        with CudaTimer(device=device, timer_name="HEAD + Loss"):
            outputs = self.lwdeter(event_frame, sparsity_mask, targets)

        loss_dict = self.criterion(outputs, targets)

        return outputs, loss_dict

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

