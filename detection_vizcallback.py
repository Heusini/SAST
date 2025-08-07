import cv2
import sys
import torch
import hydra
import numpy as np

from omegaconf import DictConfig, OmegaConf

sys.path.append(".")
from modules.utils.detection import BackboneFeatureSelector, EventReprSelector, RNNStates, Mode
from modules.utils.fetch import fetch_data_module, fetch_model_module
from data.utils.types import DataType, LstmStates, ObjDetOutput, DatasetSamplingMode, BackboneFeatures
from utils.padding import InputPadderFromShape
from models.detection.yolox.utils.boxes import postprocess
from utils.evaluation.evaluator import Evaluator
from config.modifier import dynamically_modify_train_config
from pytorch_lightning.loggers import CSVLogger
from pathlib import Path
from callbacks.detection import DetectionVizCallback

def event_data_processor(event):
    event = event.numpy()
    event = np.sum(event, axis=0)
    event = cv2.applyColorMap(cv2.normalize(event, None, 0, 255, cv2.NORM_MINMAX)
                                  .astype(np.uint8), cv2.COLORMAP_JET)
    return event

@hydra.main(config_path='config', config_name='train', version_base='1.2')
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

    # module = fetch_model_module(config=config)
    # module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config}, strict=True)

    # module.eval()
    # Get a batch (or a single sample wrapped as batch)
    data_module.setup('validate')
    val_loader = data_module.val_dataloader()

    input_padder = InputPadderFromShape(desired_hw=(height, width))

    detection_viz = DetectionVizCallback(config)

    batch = next(iter(val_loader))
    data = batch['data']
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            ev_tensor_sequence = data.get(DataType.EV_REPR)
            image_seq = data.get(DataType.IMAGE)
            sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
            is_first_sample = data[DataType.IS_FIRST_SAMPLE]
            # image = None
            # if DataType.IMAGE in data.keys():
            #     data_image = data[DataType.IMAGE]
            sequence_len = len(ev_tensor_sequence)
            batch_size = ev_tensor_sequence[0].shape[0]
            for i in range(sequence_len):
                current_labels = [l for l in sparse_obj_labels[i].sparse_object_labels_batch]
                gt_labels = [gt.get_labels_xyxy().detach().cpu().numpy() for gt in current_labels]
                output = {
                    ObjDetOutput.LABELS_PROPH: gt_labels[0],
                    ObjDetOutput.PRED_PROPH: [],
                    ObjDetOutput.IMAGE_DATA: image_seq[i],
                    ObjDetOutput.EV_REPR: ev_tensor_sequence[i],
                    ObjDetOutput.SKIP_VIZ: False,
                }

                
                detection_viz.on_validation_batch_end_custom(None, output)
            results = detection_viz.on_validation_epoch_end_custom(None)
            print(f"{len(results)=}")
            for i in range(len(results)):
                cv2.imshow("window", results[i])
                cv2.waitKey(0)


if __name__ == '__main__':
    main()
