from typing import Any, Optional, Tuple, Union, Dict
from warnings import warn
import cv2

import sys
import numpy as np
import pytorch_lightning as pl
import torch
import torch as th
import torch.nn as nn
import torch.distributed as dist
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT

from data.utils.object_labels import ObjectLabels
from data.utils.types import DataType, LstmStates, ObjDetOutput, DatasetSamplingMode, BackboneFeatures
from models.detection.yolox.utils.boxes import postprocess
from utils.evaluation.prophesee.evaluator import PropheseeEvaluator
from utils.evaluation.prophesee.io.box_loading import to_prophesee
from utils.padding import InputPadderFromShape
from .utils.detection import BackboneFeatureSelector, EventReprSelector, RNNStates, Mode, mode_2_string, \
    merge_mixed_batches
from util.scheduler import CosineLRScheduler

from utils.timers import CudaTimer

from models.detection.lwdeter.detector import LWDETERDetector


class LWDETERModule(pl.LightningModule):
    def __init__(self, full_config: DictConfig):
        super().__init__()

        self.full_config = full_config

        self.mdl_config = full_config.model
        in_res_hw = tuple(self.mdl_config.backbone.in_res_hw)
        self.input_padder = InputPadderFromShape(desired_hw=in_res_hw)

        self.mdl = LWDETERDetector(self.mdl_config)
        self.max_pool = nn.MaxPool2d(2, 2)

        self.val_losses = []

        self.mode_2_rnn_states: Dict[Mode, RNNStates] = {
            Mode.TRAIN: RNNStates(),
            Mode.VAL: RNNStates(),
            Mode.TEST: RNNStates(),
        }

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_name = self.full_config.dataset.name
        self.mode_2_hw: Dict[Mode, Optional[Tuple[int, int]]] = {}
        self.mode_2_batch_size: Dict[Mode, Optional[int]] = {}
        self.mode_2_psee_evaluator: Dict[Mode, Optional[PropheseeEvaluator]] = {}
        self.mode_2_sampling_mode: Dict[Mode, DatasetSamplingMode] = {}

        self.started_training = True

        dataset_train_sampling = self.full_config.dataset.train.sampling
        dataset_eval_sampling = self.full_config.dataset.eval.sampling
        assert dataset_train_sampling in iter(DatasetSamplingMode)
        assert dataset_eval_sampling in (DatasetSamplingMode.STREAM, DatasetSamplingMode.RANDOM)
        if stage == 'fit':  # train + val
            self.train_config = self.full_config.training
            self.train_metrics_config = self.full_config.logging.train.metrics

            if self.train_metrics_config.compute:
                self.mode_2_psee_evaluator[Mode.TRAIN] = PropheseeEvaluator(
                    dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_psee_evaluator[Mode.VAL] = PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.TRAIN] = dataset_train_sampling
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling

            for mode in (Mode.TRAIN, Mode.VAL):
                self.mode_2_hw[mode] = None
                self.mode_2_batch_size[mode] = None
            self.started_training = False
        elif stage == 'validate':
            mode = Mode.VAL
            self.mode_2_psee_evaluator[mode] = PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.VAL] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        elif stage == 'test':
            mode = Mode.TEST
            self.mode_2_psee_evaluator[mode] = PropheseeEvaluator(
                dataset=dataset_name, downsample_by_2=self.full_config.dataset.downsample_by_factor_2)
            self.mode_2_sampling_mode[Mode.TEST] = dataset_eval_sampling
            self.mode_2_hw[mode] = None
            self.mode_2_batch_size[mode] = None
        else:
            raise NotImplementedError

    # def forward(self,
    #             event_tensor: th.Tensor,
    #             previous_states: Optional[LstmStates] = None,
    #             retrieve_detections: bool = True,
    #             targets=None) \
    #         -> Tuple[Union[th.Tensor, None], Union[Dict[str, th.Tensor], None], LstmStates]:
    #     return self.mdl(x=event_tensor,
    #                     previous_states=previous_states,
    #                     retrieve_detections=retrieve_detections,
    #                     targets=targets)

    def forward(self,
                event_tensor: th.Tensor,
                previous_states: Optional[LstmStates] = None) \
            -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:

        with CudaTimer(torch.device('cuda'), "SAST"):
            output = self.mdl.forward_backbone(x=event_tensor,
                            previous_states=previous_states)[0]
        output = [output[i] for i in [1, 2, 3, 4]]
        return output
    
    def get_worker_id_from_batch(self, batch: Any) -> int:
        return batch['worker_id']

    def get_data_from_batch(self, batch: Any):
        return batch['data']

    def get_sparsity_mask(self, fpn_layer: th.Tensor, threshold = 0.7):
        tokens = torch.norm(fpn_layer, dim=1)
        min_val = tokens.amin(dim=(-2, -1), keepdim=True)
        max_val = tokens.amax(dim=(-2, -1), keepdim=True)

        tokens = (tokens - min_val) / (max_val-min_val + 1e-8)
        if self.max_pool is not None:
            tokens = self.max_pool(tokens)

        return tokens < threshold

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        batch = merge_mixed_batches(batch)
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        mode = Mode.TRAIN
        self.started_training = True
        step = self.trainer.global_step
        ev_tensor_sequence = data[DataType.EV_REPR]
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        # image_sequence = data[DataType.IMAGE]
        image_sequence = None
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

        # for o in obj_labels:
        #     print(f"{o.object_labels.dtype=}")
        # print(f"{ev_tensor_sequence[0].dtype=}")
        # sys.exit(0)
        labels_lwdetr = ObjectLabels.get_labels_as_batched_tensor(obj_label_list=obj_labels, format_='lwdetr')

        selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()

        # image_sequence = th.cat(image_sequence, dim=0)
        # image_sequence = self.input_padder.pad_tensor_ev_repr(image_sequence)

        fpn_features = self.mdl.forward_fpn(backbone_features=selected_backbone_features)

        sparsity_mask = self.get_sparsity_mask(fpn_features[0], 0.2)

        max = np.max((sparsity_mask.shape[-1], sparsity_mask.shape[-2]))
        sparsity_mask, pad = InputPadderFromShape._pad_tensor_impl(sparsity_mask, (max, max), mode='constant', value=1)
        sparsity_mask = sparsity_mask.flatten(1,2)

        predictions, loss_dict = self.mdl.forward_detect(ev_tensor_sequence, image_sequence, sparsity_mask, labels_lwdetr)
        weight_dict = self.mdl.criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_unscaled = {
        #     f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        # }
        # loss_dict_reduced_scaled = {
        #     k: v * weight_dict[k]
        #     for k, v in loss_dict_reduced.items()
        #     if k in weight_dict
        # }
        # losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        # loss_value = losses_reduced_scaled.item()


        if self.mode_2_sampling_mode[mode] in (DatasetSamplingMode.MIXED, DatasetSamplingMode.RANDOM):
            # We only want to evaluate the last batch_size samples if we use random sampling (or mixed).
            # This is because otherwise we would mostly evaluate the init phase of the sequence.
            predictions['pred_logits'] = predictions['pred_logits'][-batch_size:]
            predictions['pred_boxes'] = predictions['pred_boxes'][-batch_size:]
            predictions['enc_outputs']['pred_logits'] = predictions['enc_outputs']['pred_logits'][-batch_size:]
            obj_labels = obj_labels[-batch_size:]

        # pred_processed = postprocess(prediction=predictions,
        #                              num_classes=self.mdl_config.head.num_classes,
        #                              conf_thre=self.mdl_config.postprocess.confidence_threshold,
        #                              nms_thre=self.mdl_config.postprocess.nms_threshold)


        pred_processed = self.mdl.postprocessors['bbox'](predictions)
        pred_processed = self.to_yolox(pred_processed)
        loaded_labels_proph, yolox_preds_proph = to_prophesee(obj_labels, pred_processed)

        # print(f"{losses=}")
        assert losses is not None
        # assert 'loss' in losses

        self.smooth_loss(P, 3)

        self.trainer._logger_connector.progress_bar_metrics['SN'] = self.p_loss // 1
        self.trainer._logger_connector.progress_bar_metrics['N'] = P // 1
        self.trainer._logger_connector.progress_bar_metrics['STEP'] = self.trainer.global_step
        # For visualization, we only use the last batch_size items.
        output = {
            ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-batch_size:],
            ObjDetOutput.PRED_PROPH: yolox_preds_proph[-batch_size:],
            ObjDetOutput.EV_REPR: event_repr[-batch_size:],
            ObjDetOutput.SKIP_VIZ: False,
            'loss': losses
        }

        # Logging
        prefix = f'{mode_2_string[mode]}/'
        log_dict = {f'{prefix}{k}': v for k, v in loss_dict.items()}
        self.log_dict(log_dict, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)

        if mode in self.mode_2_psee_evaluator and len(loaded_labels_proph) > 0:
            self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
            self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)
            if self.train_metrics_config.detection_metrics_every_n_steps is not None and \
                    step > 0 and step % self.train_metrics_config.detection_metrics_every_n_steps == 0:
                self.run_psee_evaluator(mode=mode)
        return output

    def to_yolox(self, lwdeter_postprocessed):
        yolo_elements = []
        for elements in lwdeter_postprocessed:
            yolo_boxes = elements['boxes']
            yolo_labels = elements['labels'].unsqueeze(1)
            yolo_scores = elements['scores'].unsqueeze(1)
            buffer = th.zeros(yolo_boxes.shape[0], device=yolo_boxes.device).unsqueeze(1)

            yolo_format = th.cat([yolo_boxes,
                                    buffer,
                                    yolo_scores,
                                    yolo_labels], dim=1)
            yolo_elements.append(yolo_format)
        return yolo_elements

    def _val_test_step_impl(self, batch: Any, mode: Mode) -> Optional[STEP_OUTPUT]:
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        assert mode in (Mode.VAL, Mode.TEST)
        ev_tensor_sequence = data[DataType.EV_REPR]
        # image_sequence = data[DataType.IMAGE]
        image_sequence = None
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        is_first_sample = data[DataType.IS_FIRST_SAMPLE]

        self.mode_2_rnn_states[mode].reset(worker_id=worker_id, indices_or_bool_tensor=is_first_sample)

        sequence_len = len(ev_tensor_sequence)
        assert sequence_len > 0
        batch_size = len(sparse_obj_labels[0])
        if self.mode_2_batch_size[mode] is None:
            self.mode_2_batch_size[mode] = batch_size
        else:
            assert self.mode_2_batch_size[mode] == batch_size

        prev_states = self.mode_2_rnn_states[mode].get_states(worker_id=worker_id)
        obj_labels = list()
        event_repr = list()

        backbone_feature_selector = BackboneFeatureSelector()
        for tidx in range(sequence_len):
            collect_predictions = (tidx == sequence_len - 1) or \
                                  (self.mode_2_sampling_mode[mode] == DatasetSamplingMode.STREAM)
            ev_tensors = ev_tensor_sequence[tidx]
            ev_tensors = ev_tensors.to(dtype=self.dtype)
            ev_tensors = self.input_padder.pad_tensor_ev_repr(ev_tensors)
            if self.mode_2_hw[mode] is None:
                self.mode_2_hw[mode] = tuple(ev_tensors.shape[-2:])
            else:
                assert self.mode_2_hw[mode] == ev_tensors.shape[-2:]

            backbone_features, states, _ = self.mdl.forward_backbone(x=ev_tensors, previous_states=prev_states)
            prev_states = states

            current_labels = [l for l in sparse_obj_labels[tidx].sparse_object_labels_batch]

            obj_labels.extend(current_labels)
            event_repr.extend(x[0] for x in ev_tensors.split(1))
            backbone_feature_selector.add_backbone_features(backbone_features)

        self.mode_2_rnn_states[mode].save_states_and_detach(worker_id=worker_id, states=prev_states)
        if len(obj_labels) == 0:
            return {ObjDetOutput.SKIP_VIZ: True}

        labels_lwdetr = ObjectLabels.get_labels_as_batched_tensor(obj_label_list=obj_labels, format_='lwdetr')

        selected_backbone_features = backbone_feature_selector.get_batched_backbone_features()
        fpn_features = self.mdl.forward_fpn(backbone_features=selected_backbone_features)

        sparsity_mask = self.get_sparsity_mask(fpn_features[0], 0.2)

        max = np.max((sparsity_mask.shape[-1], sparsity_mask.shape[-2]))
        sparsity_mask, pad = InputPadderFromShape._pad_tensor_impl(sparsity_mask, (max, max), mode='constant', value=1)
        sparsity_mask = sparsity_mask.flatten(1,2)

        predictions, loss_dict = self.mdl.forward_detect(ev_tensor_sequence, image_sequence, sparsity_mask, labels_lwdetr)
        weight_dict = self.mdl.criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )


        pred_processed = self.mdl.postprocessors['bbox'](predictions)
        pred_processed = self.to_yolox(pred_processed)
        loaded_labels_proph, yolox_preds_proph = to_prophesee(obj_labels, pred_processed)
        # print(loaded_labels_proph)
        # For visualization, we only use the last item (per batch).
        output = {
            ObjDetOutput.LABELS_PROPH: loaded_labels_proph[-1],
            ObjDetOutput.PRED_PROPH: yolox_preds_proph[-1],
            ObjDetOutput.EV_REPR: event_repr[-1],
            ObjDetOutput.SKIP_VIZ: False,
        }

        if self.started_training:
            self.mode_2_psee_evaluator[mode].add_labels(loaded_labels_proph)
            self.mode_2_psee_evaluator[mode].add_predictions(yolox_preds_proph)
            self.val_losses.append(losses)

        return output

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode=Mode.VAL)

    def test_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        return self._val_test_step_impl(batch=batch, mode=Mode.TEST)

    def run_psee_evaluator(self, mode: Mode):
        psee_evaluator = self.mode_2_psee_evaluator[mode]
        batch_size = self.mode_2_batch_size[mode]
        hw_tuple = self.mode_2_hw[mode]
        if psee_evaluator is None:
            warn(f'psee_evaluator is None in {mode=}', UserWarning, stacklevel=2)
            return
        assert batch_size is not None
        assert hw_tuple is not None
        if psee_evaluator.has_data():
            metrics = psee_evaluator.evaluate_buffer(img_height=hw_tuple[0],
                                                     img_width=hw_tuple[1])
            assert metrics is not None

            prefix = f'{mode_2_string[mode]}/'
            step = self.trainer.global_step
            log_dict = {}
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    value = torch.tensor(v)
                elif isinstance(v, np.ndarray):
                    value = torch.from_numpy(v)
                elif isinstance(v, torch.Tensor):
                    value = v
                else:
                    raise NotImplementedError
                assert value.ndim == 0, f'tensor must be a scalar.\n{v=}\n{type(v)=}\n{value=}\n{type(value)=}'
                # put them on the current device to avoid this error: https://github.com/Lightning-AI/lightning/discussions/2529
                log_dict[f'{prefix}{k}'] = value.to(self.device)
            # Somehow self.log does not work when we eval during the training epoch.
            self.log_dict(log_dict, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            if dist.is_available() and dist.is_initialized():
                # We now have to manually sync (average the metrics) across processes in case of distributed training.
                # NOTE: This is necessary to ensure that we have the same numbers for the checkpoint metric (metadata)
                # and wandb metric:
                # - checkpoint callback is using the self.log function which uses global sync (avg across ranks)
                # - wandb uses log_metrics that we reduce manually to global rank 0
                dist.barrier()
                for k, v in log_dict.items():
                    dist.reduce(log_dict[k], dst=0, op=dist.ReduceOp.SUM)
                    if dist.get_rank() == 0:
                        log_dict[k] /= dist.get_world_size()
            if self.trainer.is_global_zero:
                # For some reason we need to increase the step by 2 to enable consistent logging in wandb here.
                # I might not understand wandb login correctly. This works reasonably well for now.
                add_hack = 2
                self.logger.log_metrics(metrics=log_dict, step=step + add_hack)

                # Determine the maximum length of the keys
                max_key_length = max(len(key) for key in log_dict.keys())
                # Print the table
                print(f"{'Metric':<{max_key_length}} | Value")
                print("-" * (max_key_length + 3) + "+" + "-" * 6)
                for key, value in log_dict.items():
                    if 'AP_S' not in key and 'AP_M' not in key and 'AP_L' not in key:
                        value = f"{value * 100:.4f}%" 
                        print(f"{key:<{max_key_length}} | {value}")

            psee_evaluator.reset_buffer()
        else:
            warn(f'psee_evaluator has not data in {mode=}', UserWarning, stacklevel=2)

    # def smooth_loss(self, step_loss, idx):
    #     if self.trainer.global_step == 0:
    #         self.loss = [0] * 5
    #         self.loss[idx] = step_loss
    #     else:
    #         self.loss[idx] = (self.loss[idx] * (self.trainer.global_step) + step_loss) / (self.trainer.global_step + 1)
    #     return self.loss[idx]

    def smooth_loss(self, step_loss, idx):
        if self.trainer.global_step == 0:
            self.iou_loss, self.conf_loss, self.cls_loss, self.p_loss = 0, 0, 0, 0
        if idx == 0:
            self.iou_loss = (self.iou_loss * (self.trainer.global_step) + step_loss) / (self.trainer.global_step + 1)
        elif idx == 1:
            self.conf_loss = (self.conf_loss * (self.trainer.global_step) + step_loss) / (self.trainer.global_step + 1)
        elif idx == 2:
            self.cls_loss = (self.cls_loss * (self.trainer.global_step) + step_loss) / (self.trainer.global_step + 1)
        elif idx == 3:
            self.p_loss = (self.p_loss * (self.trainer.global_step) + step_loss) / (self.trainer.global_step + 1)
        
    def on_train_batch_start(self, batch, batch_idx) -> None:
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.trainer._logger_connector.progress_bar_metrics['lr'] = lr
        
    def on_train_epoch_end(self) -> None:
        mode = Mode.TRAIN
        if mode in self.mode_2_psee_evaluator and \
                self.train_metrics_config.detection_metrics_every_n_steps is None and \
                self.mode_2_hw[mode] is not None:
            # For some reason PL calls this function when resuming.
            # We don't know yet the value of train_height_width, so we skip this
            self.run_psee_evaluator(mode=mode)

    def sum_losses(self, mode) -> Dict:
        accumulated_loss = {}
        prefix = f'{mode_2_string[mode]}/'
        count = 0
        for losses in self.val_losses:
            count += 1
            for k, v in losses.items():
                if isinstance(v, (int, float)):
                    value = torch.tensor(v)
                elif isinstance(v, np.ndarray):
                    value = torch.from_numpy(v)
                elif isinstance(v, torch.Tensor):
                    value = v
                else:
                    raise NotImplementedError
                assert value.ndim == 0, f'tensor must be a scalar.\n{v=}\n{type(v)=}\n{value=}\n{type(value)=}'
                if f"{prefix}{k}" not in accumulated_loss:
                    accumulated_loss[f"{prefix}{k}"] = value.to(self.device)
                else:
                    accumulated_loss[f"{prefix}{k}"] += value.to(self.device)

        for k in accumulated_loss:
            accumulated_loss[k] /= count

        return accumulated_loss

    def log_wanddb(self, log_dict):
        step = self.trainer.global_step
        if dist.is_available() and dist.is_initialized():
            # We now have to manually sync (average the metrics) across processes in case of distributed training.
            # NOTE: This is necessary to ensure that we have the same numbers for the checkpoint metric (metadata)
            # and wandb metric:
            # - checkpoint callback is using the self.log function which uses global sync (avg across ranks)
            # - wandb uses log_metrics that we reduce manually to global rank 0
            dist.barrier()
            for k, v in log_dict.items():
                dist.reduce(log_dict[k], dst=0, op=dist.ReduceOp.SUM)
                if dist.get_rank() == 0:
                    log_dict[k] /= dist.get_world_size()
        if self.trainer.is_global_zero:
            # For some reason we need to increase the step by 2 to enable consistent logging in wandb here.
            # I might not understand wandb login correctly. This works reasonably well for now.
            add_hack = 2
            self.logger.log_metrics(metrics=log_dict, step=step + add_hack)

            # Determine the maximum length of the keys
            max_key_length = max(len(key) for key in log_dict.keys())
            # Print the table
            print(f"{'Metric':<{max_key_length}} | Value")
            print("-" * (max_key_length + 3) + "+" + "-" * 6)
            for key, value in log_dict.items():
                if 'AP_S' not in key and 'AP_M' not in key and 'AP_L' not in key:
                    value = f"{value * 100:.4f}%" 
                    print(f"{key:<{max_key_length}} | {value}")


    def on_validation_epoch_end(self) -> None:
        mode = Mode.VAL
        batch_size = self.mode_2_batch_size[mode]
        if self.started_training:
            assert self.mode_2_psee_evaluator[mode].has_data()
            self.run_psee_evaluator(mode=mode)

        log_dict = self.sum_losses(mode)
        self.log_dict(log_dict, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
        self.log_wanddb(log_dict)
        # clear losses list
        self.val_losses = []

    def on_test_epoch_end(self) -> None:
        mode = Mode.TEST
        assert self.mode_2_psee_evaluator[mode].has_data()
        self.run_psee_evaluator(mode=mode)

    # def lr_scheduler_step(self, scheduler, metric, optimizer_idx):
    #     # Custom step logic
    #     scheduler.step(self.current_epoch, metric)

    def configure_optimizers(self) -> Any:
        lr = self.train_config.learning_rate
        weight_decay = self.train_config.weight_decay
        optimizer = th.optim.AdamW(filter(lambda p: p.requires_grad,self.mdl.parameters()), lr=lr, weight_decay=weight_decay)

        scheduler_params = self.train_config.lr_scheduler
        if not scheduler_params.use:
            return optimizer

        total_steps = scheduler_params.total_steps
        assert total_steps is not None
        assert total_steps > 0
        # Here we interpret the final lr as max_lr/final_div_factor.
        # Note that Pytorch OneCycleLR interprets it as initial_lr/final_div_factor:
        final_div_factor_pytorch = scheduler_params.final_div_factor / scheduler_params.div_factor
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=lr,
            div_factor=scheduler_params.div_factor,
            final_div_factor=final_div_factor_pytorch,
            total_steps=total_steps,
            pct_start=scheduler_params.pct_start,
            cycle_momentum=False,
            anneal_strategy='cos')
        # lr_scheduler = CosineLRScheduler(
        #     optimizer,
        #     t_initial=100,
        #     t_mul=1.0,
        #     lr_min=lr,
        #     decay_rate=0.1,
        #     cycle_limit=1,
        #     t_in_epochs=True,
        #     noise_range_t=None,
        #     noise_pct=0.67,
        #     noise_std=1.0,
        #     noise_seed=42,
        # )
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": 'learning_rate',
        }


        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
