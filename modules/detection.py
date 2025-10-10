from typing import Any, Optional, Tuple, Union, Dict, Callable
from types import MethodType
from warnings import warn

import sys
import numpy as np
import pytorch_lightning as pl
import torch
import torch as th
import torch.distributed as dist
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT

from data.utils.object_labels import ObjectLabels
from data.utils.types import DataType, LstmStates, ObjDetOutput, DatasetSamplingMode, BackboneFeatures, ModelOutput
from models.detection.yolox.utils.boxes import postprocess
from models.detection.yolox_extension.models.detector import YoloXDetector
from utils.evaluation.prophesee.evaluator import PropheseeEvaluator
from utils.evaluation.evaluator import Evaluator
from utils.evaluation.prophesee.io.box_loading import to_prophesee
from utils.padding import InputPadderFromShape
from .utils.detection import BackboneFeatureSelector, EventReprSelector, RNNStates, Mode, mode_2_string, \
    merge_mixed_batches

from utils.timers import CudaTimer


class Module(pl.LightningModule):
    def __init__(self, full_config: DictConfig, model: th.nn.Module, step: Callable[[Any, Any, int, Mode],Any]):
        super().__init__()

        self.full_config = full_config

        self.mdl_config = full_config.model
        in_res_hw = tuple(self.mdl_config.backbone.in_res_hw)
        self.input_padder = InputPadderFromShape(desired_hw=in_res_hw)

        self.mdl = model(self.mdl_config)

        self.val_losses = []
        self.classes = full_config.dataset.classes
        self.step = MethodType(step, self)

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
        self.mode_2_evaluator: Dict[Mode, Optional[PropheseeEvaluator]] = {}
        self.mode_2_sampling_mode: Dict[Mode, DatasetSamplingMode] = {}

        self.started_training = True


        height, width = self.mdl_config.backbone.in_res_hw

        dataset_train_sampling = self.full_config.dataset.train.sampling
        dataset_eval_sampling = self.full_config.dataset.eval.sampling
        assert dataset_train_sampling in iter(DatasetSamplingMode)
        assert dataset_eval_sampling in (DatasetSamplingMode.STREAM, DatasetSamplingMode.RANDOM)
        if stage == 'fit':  # train + val
            self.train_config = self.full_config.training
            self.train_metrics_config = self.full_config.logging.train.metrics

            if self.train_metrics_config.compute:
                self.mode_2_psee_evaluator[Mode.TRAIN] = Evaluator(self.classes, height, width)
            self.mode_2_psee_evaluator[Mode.VAL] = Evaluator(self.classes, height, width)
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
                rgb_image: th.Tensor,
                previous_states: Optional[LstmStates] = None) \
            -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:

        with CudaTimer(torch.device('cuda'), "COMPLETE_FORWARD"):
            output = self.mdl(x=event_tensor,
                            rgb_image=rgb_image,
                            previous_states=previous_states)

        return output

    def get_worker_id_from_batch(self, batch: Any) -> int:
        return batch['worker_id']

    def get_data_from_batch(self, batch: Any):
        return batch['data']


    def convert_labels_for_logging(self, predictions, obj_labels):
        # get labels in xyxy format
        gt_labels = [gt.get_labels_xyxy().detach().cpu().numpy() for gt in obj_labels]
        # boxes are in xyxy format
        pred_processed = [np.empty((0,7)) if pred is None else pred.detach().cpu().numpy() for pred in predictions]

        return pred_processed, gt_labels


    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        self.started_training = True
        mode = Mode.TRAIN
        batch = merge_mixed_batches(batch)
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)
        step = self.trainer.global_step

        batch_size = self.full_config.batch_size.train
        model_output = self.step(data, batch_idx, mode, worker_id)

        predictions = model_output[ModelOutput.PREDICTIONS]
        obj_labels = model_output[ModelOutput.GROUND_TRUTHS]
        losses = model_output[ModelOutput.LOSSES]

        if self.mode_2_sampling_mode[mode] in (DatasetSamplingMode.MIXED, DatasetSamplingMode.RANDOM):
            # We only want to evaluate the last batch_size samples if we use random sampling (or mixed).
            # This is because otherwise we would mostly evaluate the init phase of the sequence.
            predictions = predictions[-batch_size:]
            obj_labels = obj_labels[-batch_size:]

        pred_processed, gt_processed = self.convert_labels_for_logging(predictions, obj_labels)
        # print(f"{pred_processed=}")
        # print(f"{gt_processed=}")

        assert losses is not None
        assert 'loss' in losses

        # self.smooth_loss(P, 3)

        # self.trainer._logger_connector.progress_bar_metrics['SN'] = self.p_loss // 1
        # self.trainer._logger_connector.progress_bar_metrics['N'] = P // 1
        self.trainer._logger_connector.progress_bar_metrics['STEP'] = self.trainer.global_step

        # For visualization, we only use the last batch_size items.
        sparsity_mask = model_output.get(ModelOutput.SPARSITY_MASK)
        if sparsity_mask is not None:
            sparsity_mask = sparsity_mask.detach().cpu().numpy()
            sparsity_mask = sparsity_mask[-batch_size:]

        event_repr = model_output.get(ModelOutput.EVENT_DATA)
        if event_repr is not None:
            event_repr = event_repr[-batch_size:]
        image_data = model_output.get(ModelOutput.IMAGE_DATA)
        if image_data is not None:
            image_data = image_data[-batch_size:]

        output = {
            ObjDetOutput.LABELS_PROPH: gt_processed[-batch_size:],
            ObjDetOutput.PRED_PROPH: pred_processed[-batch_size:],
            ObjDetOutput.SPARSITY_MASK: sparsity_mask,
            ObjDetOutput.EV_REPR: event_repr,
            ObjDetOutput.IMAGE_DATA: image_data,
            ObjDetOutput.SKIP_VIZ: False,
            'loss': losses['loss']
        }

        # Logging
        prefix = f'{mode_2_string[mode]}/'
        log_dict = {f'{prefix}{k}': v for k, v in losses.items()}
        self.log_dict(log_dict, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)

        if mode in self.mode_2_psee_evaluator:
            self.mode_2_psee_evaluator[mode].add_labels(gt_processed)
            self.mode_2_psee_evaluator[mode].add_predictions(pred_processed)
            if self.train_metrics_config.detection_metrics_every_n_steps is not None and \
                    step > 0 and step % self.train_metrics_config.detection_metrics_every_n_steps == 0:
                self.run_psee_evaluator(mode=mode)
        return output

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        mode = Mode.VAL
        data = self.get_data_from_batch(batch)
        worker_id = self.get_worker_id_from_batch(batch)

        model_output = self.step(data, batch_idx, mode, worker_id)

        predictions = model_output[ModelOutput.PREDICTIONS]
        obj_labels = model_output[ModelOutput.GROUND_TRUTHS]
        losses = model_output[ModelOutput.LOSSES]

        pred_processed, gt_processed = self.convert_labels_for_logging(predictions, obj_labels)

        sparsity_mask = model_output.get(ModelOutput.SPARSITY_MASK)
        if sparsity_mask is not None:
            sparsity_mask = sparsity_mask.detach().cpu().numpy()
            sparsity_mask = sparsity_mask[-1]

        event_repr = model_output.get(ModelOutput.EVENT_DATA)
        if event_repr is not None:
            event_repr = event_repr[-1]
        image_data = model_output.get(ModelOutput.IMAGE_DATA)
        if image_data is not None:
            image_data = image_data[-1]

        output = {
            ObjDetOutput.LABELS_PROPH: gt_processed[-1],
            ObjDetOutput.PRED_PROPH: pred_processed[-1],
            ObjDetOutput.EV_REPR: event_repr,
            ObjDetOutput.SPARSITY_MASK: sparsity_mask,
            ObjDetOutput.IMAGE_DATA: image_data,
            ObjDetOutput.SKIP_VIZ: False,
        }

        if self.started_training:
            self.mode_2_psee_evaluator[mode].add_labels(gt_processed)
            self.mode_2_psee_evaluator[mode].add_predictions(pred_processed)
            self.val_losses.append(losses)

        return output

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
            # print("has_data")
            metrics = psee_evaluator.evaluate_buffer()
            # print(f"{metrics=}")
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

    def log_wandb(self, log_dict):
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
        self.log_wandb(log_dict)
        # clear losses list
        self.val_losses = []

    def on_test_epoch_end(self) -> None:
        mode = Mode.TEST
        assert self.mode_2_psee_evaluator[mode].has_data()
        self.run_psee_evaluator(mode=mode)

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
            anneal_strategy='linear')
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
            "name": 'learning_rate',
        }

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}
