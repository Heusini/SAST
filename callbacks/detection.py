import cv2
from enum import Enum, auto
from typing import Any

import torch
from einops import rearrange
from omegaconf import DictConfig

from data.utils.types import ObjDetOutput
from loggers.wandb_logger import WandbLogger
from utils.evaluation.prophesee.visualize.vis_utils import LABELMAP_GEN1, LABELMAP_GEN4_SHORT, draw_bboxes
from .viz_base import VizCallbackBase
import bbox_visualizer as bbv
import numpy as np

RED=(255, 0, 0)
GREEN=(0, 255, 0)
BLUE=(0, 0, 255)

class DetectionVizEnum(Enum):
    EV_IMG = auto()
    LABEL_IMG_PROPH = auto()
    PRED_IMG_PROPH = auto()
    LABEL_EVENT_PROPH = auto()
    PRED_EVENT_PROPH = auto()

class DetectionVizCallback(VizCallbackBase):
    def __init__(self, config: DictConfig):
        super().__init__(config=config, buffer_entries=DetectionVizEnum)

        dataset_name = config.dataset.name
        self.label_map = config.dataset.classes

        # fixed for now maybe add to config
        self.confidence_threshold = 0.3

    def get_bbox_text(self, bbox):
        class_name = self.label_map[int(bbox[4]) % len(self.label_map)]
        output = f"{class_name}"
        if len(bbox) > 5:
            score = bbox[5]
            output = f"{output}: {score:.2f}"
        return output

    # This all could be improved if we would use a format with named indexes
    def filter(self, bbox):
        # we check if the box is a prediction box or ground truth by the length
        if len(bbox) > 5:
            # we expect the yolo format where the 5th entry is the confidence
            # if the bbox confidence is smaller than the threshold it gets filtered
            return bbox[5] <= self.confidence_threshold
        else:
            return False

    def draw_filtered_bboxes(self, image, bboxes, color):
        for box in bboxes:
            if box is None or len(box) == 0 or self.filter(box):
                continue
            bb = box.copy().astype(np.int32)
            bb = bb[:4]
            image = bbv.draw_rectangle(image, bb, bbox_color=color, thickness=1)
            # bbox_txt = self.get_bbox_text(box)
            # image = bbv.add_label(image, bbox_txt, bb, text_bg_color=color, size=0.2,top=True)
        return image

    def draw_bboxes(self, image, bboxes, color):
        for box in bboxes:
            if box is None or len(box) == 0:
                continue
            bb = box.copy().astype(np.int32)
            bb = bb[:4]
            image = bbv.draw_rectangle(image, bb, bbox_color=color, thickness=1)
            # bbox_txt = self.get_bbox_text(box)
            # image = bbv.add_label(image, bbox_txt, bb, text_bg_color=color, size=0.2,top=True)
        return image

    def apply_sparsity_mask(self, img, sparsity_mask, alpha = 0.4): 
        # a lot of magic numbers -> change somehow
        sp_mask = sparsity_mask.reshape(24,40)
        sp_mask = np.kron(sp_mask, np.ones((16, 16), dtype=sparsity_mask.dtype))
        mask_img = np.zeros_like(img)
        # mask_img[:, :, 3][sp_mask] = 255
        mask_img[:, :, 0][sp_mask] = 255
        img = cv2.addWeighted(mask_img, alpha, img, 1-alpha, 0)

        return img

    def draw_on_image(self, img, sparsity_mask, labels, label_color):
        img = img.copy()
        if sparsity_mask is not None:
            img = self.apply_sparsity_mask(img, sparsity_mask)
        img = self.draw_filtered_bboxes(img, labels, color=label_color)
        return img

        

    def on_train_batch_end_custom(self,
                                  logger: WandbLogger,
                                  outputs: Any,
                                  batch: Any,
                                  log_n_samples: int,
                                  global_step: int) -> None:
        if outputs is None:
            # If we tried to skip the training step (not supported in DDP in PL, atm)
            return
        ev_tensors = outputs.get(ObjDetOutput.EV_REPR)
        image_seq = outputs.get(ObjDetOutput.IMAGE_DATA)
        sparsity_mask = outputs.get(ObjDetOutput.SPARSITY_MASK)
        predictions = outputs[ObjDetOutput.PRED_PROPH]
        labels = outputs[ObjDetOutput.LABELS_PROPH]
        num_samples = len(labels)
        assert num_samples > 0
        log_n_samples = min(num_samples, log_n_samples)

        merged_imgs = []
        captions = []
        start_idx = num_samples - 1
        end_idx = start_idx - log_n_samples
        # for sample_idx in range(log_n_samples):
        for sample_idx in range(start_idx, end_idx, -1):
            predictions_idx = predictions[sample_idx]
            labels_idx = labels[sample_idx]
            mask = None
            merged_events = None
            merged_rgbs = None
            if sparsity_mask is not None:
                mask = sparsity_mask[sample_idx]
            if ev_tensors is not None:
                ev_img = self.ev_repr_to_img(ev_tensors[sample_idx].cpu().numpy())
                ev_pred_img = self.draw_on_image(ev_img, mask, predictions_idx, BLUE)

                ev_label_img = self.draw_on_image(ev_img, mask, labels_idx, GREEN)
                merged_events = rearrange([ev_pred_img, ev_label_img], 'pl H W C -> (pl H) W C', pl=2, C=3)

            if image_seq is not None:
                image = image_seq[sample_idx].squeeze(0).permute(1,2,0)
                image = image.cpu().numpy()
                image *= 255
                image = image.astype(np.uint8)
                prediction_img = self.draw_on_image(image, mask, predictions_idx, BLUE)

                label_img = self.draw_on_image(image, mask, labels_idx, GREEN)
                merged_rgbs = rearrange([prediction_img, label_img], 'pl H W C -> (pl H) W C', pl=2, C=3)
            
            merger = None
            if merged_events is not None and merged_rgbs is not None:
                merger = np.hstack([merged_events, merged_rgbs])
            elif merged_events is not None:
                merger = merged_events
            else:
                merger = merged_rgbs

            merged_imgs.append(merger)
            captions.append(f'sample_{sample_idx}')

        logger.log_images(key='train/predictions',
                          images=merged_imgs,
                          caption=captions,
                          step=global_step)



    def on_validation_batch_end_custom(self, batch: Any, outputs: Any):
        if outputs[ObjDetOutput.SKIP_VIZ]:
            return
        ev_tensor = outputs.get(ObjDetOutput.EV_REPR)
        image = outputs.get(ObjDetOutput.IMAGE_DATA)
        sparsity_mask = outputs.get(ObjDetOutput.SPARSITY_MASK)

        predictions_proph = outputs[ObjDetOutput.PRED_PROPH]
        labels_proph = outputs[ObjDetOutput.LABELS_PROPH]
        if ev_tensor is not None:
            ev_img = self.ev_repr_to_img(ev_tensor.cpu().numpy())

            pred_ev_img = self.draw_on_image(ev_img, sparsity_mask, predictions_proph, BLUE)
            self.add_to_buffer(DetectionVizEnum.PRED_EVENT_PROPH, pred_ev_img)

            pred_label_img = self.draw_on_image(ev_img, sparsity_mask, labels_proph, GREEN)
            self.add_to_buffer(DetectionVizEnum.LABEL_EVENT_PROPH, pred_label_img)

        if image is not None:
            image = image.squeeze(0).permute(1,2,0)
            image = image.cpu().numpy()
            image *= 255
            image = image.astype(np.uint8)
            
            prediction_img = self.draw_on_image(image, sparsity_mask, predictions_proph, BLUE)
            self.add_to_buffer(DetectionVizEnum.PRED_IMG_PROPH, prediction_img)

            label_img = self.draw_on_image(image, sparsity_mask, labels_proph, GREEN)
            self.add_to_buffer(DetectionVizEnum.LABEL_IMG_PROPH, label_img)

    def on_validation_epoch_end_custom(self, logger: WandbLogger):
        pred_imgs = self.get_from_buffer(DetectionVizEnum.PRED_IMG_PROPH)
        label_imgs = self.get_from_buffer(DetectionVizEnum.LABEL_IMG_PROPH)
        # assert len(pred_imgs) == len(label_imgs)
        pred_evs = self.get_from_buffer(DetectionVizEnum.PRED_EVENT_PROPH)
        label_evs = self.get_from_buffer(DetectionVizEnum.LABEL_EVENT_PROPH)
        # assert len(pred_evs) == len(label_evs)
        all_img = []
        merged_img = []
        captions = []
        for idx, (pred_img, label_img) in enumerate(zip(pred_imgs, label_imgs)):
            merged_img.append(rearrange([pred_img, label_img], 'pl H W C -> (pl H) W C', pl=2, C=3))
            captions.append(f'sample_{idx}')

        merged_evs = []
        for idx, (pred_ev, label_ev) in enumerate(zip(pred_evs, label_evs)):
            merged_evs.append(rearrange([pred_ev, label_ev], 'pl H W C -> (pl H) W C', pl=2, C=3))
            captions.append(f'sample_{idx}')

        if len(merged_evs) > 0 and len(merged_img) > 0:
            for ev, img in zip(merged_evs, merged_img):
                new_img = np.hstack([ev, img])
                all_img.append(new_img)
        elif len(merged_evs) > 0:
            all_img = merged_evs
        else:
            all_img = merged_img

        captions = captions[:max(len(merged_evs), len(merged_img))]
        logger.log_images(key='val/predictions',
                          images=all_img,
                          caption=captions)
