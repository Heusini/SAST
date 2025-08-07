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
    ev_tensor_sequence = data[DataType.EV_REPR]
    sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
    is_first_sample = data[DataType.IS_FIRST_SAMPLE]
    token_mask_sequence = data.get(DataType.TOKEN_MASK, None)

    self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

    sequence_len = len(ev_tensor_sequence)
    assert sequence_len > 0
    batch_size = len(sparse_obj_labels[0])
    if self.mode_2_batch_size[mode] is None:
        self.mode_2_batch_size[mode] = batch_size
    else:
        assert self.mode_2_batch_size[mode] == batch_size

    prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
    backbone_feature_selector = BackboneFeatureSelector()
    obj_labels = list()
    event_repr = list()
    P = 0
    for tidx in range(sequence_len):
        ev_tensors = ev_tensor_sequence[tidx]
        ev_tensors = ev_tensors.to(dtype=self.dtype)
        ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
        if token_mask_sequence is not None:
            token_masks = self.input_padder.pad_token_mask(token_mask=token_mask_sequence[tidx])
        else:
            token_masks = None

        if self.mode_2_hw[mode] is None:
            self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
        else:
            assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

        backbone_features, states, p = self.mdl.forward_backbone(x=ev_tensors,
                                                              previous_states=prev_states,
                                                              token_mask=token_masks)
        P += sum(p) / sequence_len
        prev_states = states

        current_labels = [l for l in sparse_obj_labels[tidx].sparse_object_labels_batch]
        obj_labels.extend(current_labels)
        event_repr.extend(x[0] for x in ev_tensors.split(1))
        backbone_feature_selector.add_backbone_features(backbone_features)

    self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)
    # Batch the backbone features and labels to parallelize the detection code.
    # selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()

    selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()
    fpn_features = self.mdl.forward_fpn(backbone_features=selected_backbone_features)

    # we return tokens without threshold right now maybe fix
    sparsity_mask = self.mdl.get_sparsity_mask(fpn_features[0], 0.2)
    sparsity_mask = sparsity_mask > 0.15
    # max = np.max((sparsity_mask.shape[-1], sparsity_mask.shape[-2]))
    # sparsity_mask = th.ones_like(sparsity_mask).to(bool)
    # sparsity_mask, pad = InputPadderFromShape._pad_tensor_impl(sparsity_mask, (max, max), mode='constant', value=False)
    sparsity_mask = sparsity_mask.flatten(1,2)
    # print(f"{sparsity_mask.shape=}")

    image_sequence = None
    labels_lwdetr = ObjectLabels.get_labels_as_batched_tensor(obj_label_list=obj_labels, format_='lwdetr')
    predictions, losses = self.mdl.forward_detect(ev_tensor_sequence, image_sequence, sparsity_mask, labels_lwdetr)

    predictions = convert_predictions(predictions)
    output = {
            ModelOutput.PREDICTIONS: predictions,
            ModelOutput.LOSSES: losses,
            ModelOutput.GROUND_TRUTHS: obj_labels,
            ModelOutput.EVENT_DATA: event_repr[-batch_size:],
            ModelOutput.SPARSITY_MASK: sparsity_mask[-batch_size:],
    }

    return output
