import sys
import torch as th
import numpy as np
from typing import Any
from modules.utils.detection import Mode
from modules.utils.detection import Mode, BackboneFeatureSelector
from data.utils.object_labels import ObjectLabels
from data.utils.types import DataType, ModelOutput
from utils.padding import InputPadderFromShape


def convert_predictions(predictions):
    ''' Convert predictions from lwdeter to xyxy obj_confidence score category_id''' 

    new_predictions = []
    for pred in predictions:
        boxes = pred['boxes']
        scores = pred['scores']
        labels = pred['labels']

        # is not used but yolox returns it and location is important in 
        # to_coco_format
        obj_confidence = th.ones_like(labels)
        new_preds = th.hstack([boxes, 
                               obj_confidence.unsqueeze(-1), 
                               scores.unsqueeze(-1), 
                               labels.unsqueeze(-1)])
        new_predictions.append(new_preds)

    return new_predictions

def step(self, data: Any, batch_idx: int, mode: Mode, worker_id):
    step = self.trainer.global_step
    image_sequence = data[DataType.IMAGE]
    sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
    is_first_sample = data[DataType.IS_FIRST_SAMPLE]
    token_mask_sequence = data.get(DataType.TOKEN_MASK, None)

    sequence_len = len(image_sequence)
    assert sequence_len > 0
    batch_size = len(sparse_obj_labels[0])
    if self.mode_2_batch_size[mode] is None:
        self.mode_2_batch_size[mode] = batch_size
    else:
        assert self.mode_2_batch_size[mode] == batch_size


    image_sequence = th.cat(image_sequence, dim=0)
    if self.mode_2_hw[mode] is None:
        self.mode_2_hw[mode] = tuple(image_sequence[0].shape[-2:])
    obj_labels = [l for labels in sparse_obj_labels for l in labels.sparse_object_labels_batch]

    labels_lwdetr = []
    count = 1
    for labels in obj_labels:
        obj_labels_coco = labels.get_labels_lwdetr_normalized(image_sequence[0].shape[-2:])
        obj_labels_coco["image_id"] = count
        labels_lwdetr.append(obj_labels_coco)
        count += 1

    predictions, losses = self.mdl.forward_detect(image_sequence, labels_lwdetr)

    predictions = convert_predictions(predictions)
    output = {
            ModelOutput.PREDICTIONS: predictions,
            ModelOutput.LOSSES: losses,
            ModelOutput.GROUND_TRUTHS: obj_labels,
            ModelOutput.IMAGE_DATA: image_sequence,
    }

    return output
