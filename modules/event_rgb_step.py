import torch as th
from typing import Any
from modules.utils.detection import Mode, BackboneFeatureSelector
from data.utils.object_labels import ObjectLabels
from data.utils.types import DataType
from models.detection.yolox.utils.boxes import postprocess
from data.utils.types import ModelOutput


def step(self, data: Any, batch_idx: int, mode: Mode, worker_id):
    step = self.trainer.global_step
    ev_tensor_sequence = data[DataType.EV_REPR]
    sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
    image_sequence = data[DataType.IMAGE]
    is_first_sample = data[DataType.IS_FIRST_SAMPLE]
    token_mask_sequence = data.get(DataType.TOKEN_MASK, None)

    self.mode_2_rnn_states[mode].reset(
        worker_id=worker_id, indices_or_bool_tensor=is_first_sample
    )

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
            token_masks = self.input_padder.pad_token_mask(
                token_mask=token_mask_sequence[tidx]
            )
        else:
            token_masks = None

        if self.mode_2_hw[mode] is None:
            self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
        else:
            assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

        backbone_features, states, p = self.mdl.forward_backbone(
            x=ev_tensors, previous_states=prev_states, token_mask=token_masks
        )
        P += sum(p) / sequence_len
        prev_states = states

        current_labels = [l for l in sparse_obj_labels[tidx].sparse_object_labels_batch]
        obj_labels.extend(current_labels)
        event_repr.extend(x[0] for x in ev_tensors.split(1))
        backbone_feature_selector.add_backbone_features(backbone_features)

    self.mode_2_rnn_states[mode].save_states_and_detach(
        worker_id=worker_id, states=prev_states
    )
    # Batch the backbone features and labels to parallelize the detection code.
    # selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()

    labels_yolox = ObjectLabels.get_labels_as_batched_tensor(
        obj_label_list=obj_labels, format_="yolox"
    )
    labels_yolox = labels_yolox.to(dtype=self.dtype)

    selected_backbone_features = (
        backbone_feature_selector.get_batched_backbone_features()
    )

    image_sequence = th.cat(image_sequence, dim=0)
    image_sequence = self.input_padder.pad_tensor_ev_repr(image_sequence)
    predictions, losses = self.mdl.forward_detect(
        backbone_features=selected_backbone_features,
        rgb_image=image_sequence,
        targets=labels_yolox,
    )

    predictions = postprocess(
        prediction=predictions,
        num_classes=self.mdl_config.head.num_classes,
        conf_thre=self.mdl_config.postprocess.confidence_threshold,
        nms_thre=self.mdl_config.postprocess.nms_threshold,
    )

    output = {
        ModelOutput.PREDICTIONS: predictions,
        ModelOutput.LOSSES: losses,
        ModelOutput.GROUND_TRUTHS: obj_labels,
        ModelOutput.IMAGE_DATA: image_sequence,
        ModelOutput.EVENT_DATA: event_repr,
    }

    return output
