import sys
import torch as th
import numpy as np
from typing import Any
from modules.utils.detection import Mode
from modules.utils.detection import Mode, BackboneFeatureSelector
from data.utils.object_labels import ObjectLabels
from data.utils.types import DataType, ModelOutput
from utils.padding import InputPadderFromShape
from models.detection.yolox.utils.boxes import postprocess


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
    sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
    image_sequence = data[DataType.IMAGE]
    # self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)
    if self.mode_2_hw[mode] is None:
        self.mode_2_hw[mode] = tuple(image_sequence[0].shape[-2:])

    sequence_len = len(image_sequence)
    assert sequence_len > 0
    batch_size = len(sparse_obj_labels[0])
    if self.mode_2_batch_size[mode] is None:
        self.mode_2_batch_size[mode] = batch_size
    else:
        assert self.mode_2_batch_size[mode] == batch_size

    # object_labels = []
    # for labels in sparse_obj_labels:
    #     current_labels = [l for l in labels.sparse_object_labels_batch]
    #     object_labels.extend(current_labels)
    # this is the same as this ^ but a list comprehension because we want to be fancy
    obj_labels = [l for labels in sparse_obj_labels for l in labels.sparse_object_labels_batch]

    labels_yolox = ObjectLabels.get_labels_as_batched_tensor(obj_label_list=obj_labels, format_='yolox')
    labels_yolox = labels_yolox.to(dtype=self.dtype)

    image_sequence = th.cat(image_sequence, dim=0)
    image_sequence = self.input_padder.pad_tensor_ev_repr(image_sequence)
    predictions, losses = self.mdl.forward_detect(image_sequence, labels_yolox)

    predictions = postprocess(prediction=predictions,
                                 num_classes=self.mdl_config.head.num_classes,
                                 conf_thre=self.mdl_config.postprocess.confidence_threshold,
                                 nms_thre=self.mdl_config.postprocess.nms_threshold)

    output = {
            ModelOutput.PREDICTIONS: predictions,
            ModelOutput.LOSSES: losses,
            ModelOutput.GROUND_TRUTHS: obj_labels,
            ModelOutput.IMAGE_DATA: image_sequence,
            }
    return output
