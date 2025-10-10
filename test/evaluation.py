import cv2
import sys
import torch
import hydra
import numpy as np

from omegaconf import DictConfig, OmegaConf

sys.path.append(".")
from data.utils.types import DataType, LstmStates, ObjDetOutput, DatasetSamplingMode, BackboneFeatures
from utils.padding import InputPadderFromShape
from models.detection.yolox.utils.boxes import postprocess
from utils.evaluation.evaluator import Evaluator
from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_data_module, fetch_model_module
from pytorch_lightning.loggers import CSVLogger
from pathlib import Path
from modules.utils.detection import BackboneFeatureSelector, EventReprSelector, RNNStates, Mode

def event_data_processor(event):
    event = event.numpy()
    event = np.sum(event, axis=0)
    event = cv2.applyColorMap(cv2.normalize(event, None, 0, 255, cv2.NORM_MINMAX)
                                  .astype(np.uint8), cv2.COLORMAP_JET)
    return event

@hydra.main(config_path='../config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    mdl_config = config.model
    classes = config.dataset.classes
    height, width = mdl_config.backbone.in_res_hw

    gpus = config.hardware.gpus
    assert isinstance(gpus, int), 'no more than 1 GPU supported'
    gpus = [gpus]
    data_module = fetch_data_module(config=config)
    logger = CSVLogger(save_dir='./validation_logs')
    ckpt_path = Path(config.checkpoint)

    module = fetch_model_module(config=config)
    module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config}, strict=True)

    module.eval()
    # Get a batch (or a single sample wrapped as batch)
    data_module.setup('validate')
    val_loader = data_module.val_dataloader()

    input_padder = InputPadderFromShape(desired_hw=(height, width))
    evaluator = Evaluator(classes, height, width)

    batch = next(iter(val_loader))
    data = batch['data']
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            ev_tensor_sequence = data[DataType.EV_REPR]
            sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
            is_first_sample = data[DataType.IS_FIRST_SAMPLE]
            # image = None
            # if DataType.IMAGE in data.keys():
            #     data_image = data[DataType.IMAGE]
            token_mask_sequence = data.get(DataType.TOKEN_MASK, None)
            sequence_len = len(ev_tensor_sequence)
            batch_size = ev_tensor_sequence[0].shape[0]
            ev_repr_selector = EventReprSelector()
            backbone_feature_selector = BackboneFeatureSelector()
            obj_labels = list()
            event_repr = list()
            for tidx in range(sequence_len):
                ev_tensors = ev_tensor_sequence[tidx]
                ev_tensors = input_padder.pad_tensor_ev_repr(ev_tensors)
                # image = input_padder.pad_tensor_ev_repr(data_image[tidx])
                bboxes = sparse_obj_labels[tidx][0]
                new_bbs = None
                if bboxes:
                    new_bb = bboxes.object_labels[:, 1:5]
                    new_bbs = np.vstack(new_bb)
                backbone_features, _, _ = module.mdl.forward_backbone(x=ev_tensors)

                current_labels = [l for l in sparse_obj_labels[tidx].sparse_object_labels_batch]
                obj_labels.extend(current_labels)
                event_repr.extend(x[0] for x in ev_tensors.split(1))
                backbone_feature_selector.add_backbone_features(backbone_features)



            features = backbone_feature_selector.get_batched_backbone_features()
            output, _ = module.mdl.forward_detect(features)
            pred_processed = postprocess(prediction=output,
                                         num_classes=mdl_config.head.num_classes,
                                         conf_thre=mdl_config.postprocess.confidence_threshold,
                                         nms_thre=mdl_config.postprocess.nms_threshold)
            gt_labels = [gt.get_labels_xyxy().detach().cpu().numpy() for gt in obj_labels]
            # boxes are in xyxy format
            pred_processed = [np.empty((0,7)) if pred is None else pred.detach().cpu().numpy() for pred in pred_processed]
            evaluator.add_labels(gt_labels)
            evaluator.add_predictions(pred_processed)
            out = evaluator.evaluate_buffer()
            img = event_data_processor(ev_tensors[0])

            # maybe add bounding boxes to this one image
            cv2.imshow("test", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            for k in out.keys():
                print(f"{k}: {out[k]:.2f}")
if __name__ == '__main__':
    main()
