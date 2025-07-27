import os
import threading

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path

import torch
import torch.nn as nn
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')

import cv2
import sys
import hydra
import numpy as np
import bbox_visualizer as bbv
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelSummary

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_data_module, fetch_model_module
from data.utils.types import DataType, LstmStates, ObjDetOutput, DatasetSamplingMode, BackboneFeatures
from utils.padding import InputPadderFromShape
from models.detection.yolox.utils.boxes import postprocess

from queue import Queue
import matplotlib.pyplot as plt
from visualizer import Visualizer
from pycocotools.coco import COCO
try:
    coco_eval_type = 'cpp-based'
    from detectron2.evaluation.fast_eval_api import COCOeval_opt as COCOeval
except ImportError:
    coco_eval_type = 'python-based'
    from pycocotools.cocoeval import COCOeval

print(f'Using {coco_eval_type} detection evaluation')
HEIGHT = 384
WIDTH = 640

def to_coco_format(gts, detections, categories=["drone"], height=HEIGHT, width=WIDTH):
    """
    utilitary function producing our data in a COCO usable format
    """
    annotations = []
    results = []
    images = []

    categories = [{"id": id + 1, "name": class_name, "supercategory": "none"}
                  for id, class_name in enumerate(categories)]
    # to dictionary
    assert len(gts) == len(detections)
    for image_id, (gt, pred) in enumerate(zip(gts, detections)):
        im_id = image_id + 1

        images.append(
            {"date_captured": "2019",
             "file_name": "n.a",
             "id": im_id,
             "license": 1,
             "url": "",
             "height": height,
             "width": width})

        for bbox in gt:
            bbox = bbox.numpy()
            x1, y1 = bbox[0], bbox[1]
            w, h = bbox[2], bbox[3]
            area = w * h
            annotation = {
                "area": float(area),
                "iscrowd": False,
                "image_id": im_id,
                "bbox": [x1, y1, w, h],
                "category_id": int(bbox[4]) + 1,
                "id": len(annotations) + 1
            }
            annotations.append(annotation)

        if pred is not None:
            for bbox in pred:
                bbox = bbox.numpy()
                x1, y1 = bbox[0], bbox[1]
                w, h = bbox[2]-x1, bbox[3]-y1
                image_result = {
                    'image_id': im_id,
                    'category_id': int(bbox[6]) + 1,
                    'score': float(bbox[5]),
                    'bbox': [x1, y1, w, h],
                }
            results.append(image_result)

    dataset = {"info": {},
               "licenses": [],
               "type": 'instances',
               "images": images,
               "annotations": annotations,
               "categories": categories}
    return dataset, results

def event_data_processor(event):
    event = event.numpy()
    event = np.sum(event, axis=0)
    event = cv2.applyColorMap(cv2.normalize(event, None, 0, 255, cv2.NORM_MINMAX)
                                  .astype(np.uint8), cv2.COLORMAP_JET)
    return event

def image_data_processor(image):
    image = image.permute(1,2,0).numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image * 255
    image = image.astype(np.uint8)
    return image

def bbox_xywh_to_xyxy(bbox):
    new_bb = bbox
    new_bb[:, 2:] += new_bb[:, :2]
    return new_bb

def bbox_processor(bboxes):
    out_bbs = []
    for bbox in bboxes:
        new_box = bbox.clone()
        new_box = new_box[:, :4].numpy()
        formated_bbox = bbox_xywh_to_xyxy(new_box).astype(np.int32)
        out_bbs.append(formated_bbox)
    return out_bbs



class PredProcessor():
    def __init__(self, mdl_config):
        self.num_classes=mdl_config.head.num_classes
        self.conf_thre=mdl_config.postprocess.confidence_threshold
        self.nms_thre=mdl_config.postprocess.nms_threshold

    def postprocess(self, preds):
        preds = postprocess(prediction=preds,
                                     num_classes=self.num_classes,
                                     conf_thre=self.conf_thre,
                                     nms_thre=self.nms_thre)
        return preds

    def pred_processor(self, preds):
        preds = self.postprocess(preds)
        out_preds = []
        for pred in preds:
            if pred is not None:
                pred = pred.numpy()
                pred = pred[:, :4].astype(np.int32)
            out_preds.append(pred)
        return out_preds
def draw_and_display_threaded(queue):
    while True:
        try:
            image = queue.get(block=True)
            cv2.imshow("frame", image)
            cv2.waitKey(0)
        except Exception as e:
            print(f"Error in displaying: {e}")
            cv2.destroyAllWindows()


def process_coco(dataset, results, imgids):
    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    coco_pred = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = np.arange(1, imgids + 1, dtype=int)

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()



def run_event_rgb(module, data_loader, config, queue):
    in_res_hw = tuple(config.model.backbone.in_res_hw)
    input_padder = InputPadderFromShape(desired_hw=in_res_hw)
    mdl_config = config.model

    max_pool = nn.MaxPool2d(2,2)
    predictions = list()
    ground_truths = list()
    batch_size = config.batch_size.eval
    pred_processor = PredProcessor(mdl_config)
    visualizer = Visualizer(batch_size, 
                            [event_data_processor, image_data_processor],
                            bbox_processor, pred_processor.pred_processor)
    ground_truths = []
    predictions = []
    for batch in data_loader:
        data = batch['data']
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                ev_tensor_sequence = data[DataType.EV_REPR]
                sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
                is_first_sample = data[DataType.IS_FIRST_SAMPLE]
                token_mask_sequence = data.get(DataType.TOKEN_MASK, None)
                image_sequence = data[DataType.IMAGE]

                sequence_len = len(ev_tensor_sequence)
                batch_images = []
                for tidx in range(sequence_len):
                    ev_tensors = ev_tensor_sequence[tidx]
                    ev_tensors = input_padder.pad_tensor_ev_repr(ev_tensors)
                    image = input_padder.pad_tensor_ev_repr(image_sequence[tidx])
                    bboxes = [box.get_labels() 
                              for box in sparse_obj_labels[tidx]
                              .sparse_object_labels_batch]

                    ground_truths.extend([box.clone() for box in bboxes])

                    preds, _, _ = module.mdl.backbone(ev_tensors)
                    preds = module.mdl.fpn(preds)
                    rgb_preds = module.mdl.rgb_fpn(image)
                    features = []
                    for f, r in zip(preds, rgb_preds):
                        intermediate_features = torch.add(f, r)
                        features.append(intermediate_features)

                    output, _ = module.mdl.yolox_head(features)
                    processed_preds = pred_processor.postprocess(output.clone())
                    predictions.extend(processed_preds)

                    # print(pred_processed)
                    # print(new_bbs)
                    images = visualizer.render_image([ev_tensors, image], [True, True], bboxes, output)
                    batch_images.append(images)
                for b in range(batch_size):
                    for s in range(sequence_len):
                        queue.put(batch_images[s][b])

                sum = np.sum(np.asarray([len(gt) for gt in ground_truths]))
                if sum > 0:
                    dataset, results = to_coco_format(ground_truths, predictions)
                    print(f"{dataset['annotations']=}")
                    print(f"{results=}")
                    process_coco(dataset, results, len(ground_truths))

@hydra.main(config_path='config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

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

    queue = Queue(maxsize=200)
    draw_thread = threading.Thread(target=draw_and_display_threaded, args=(queue,),)
    draw_thread.daemon = True
    draw_thread.start()
    run_event_rgb(module, val_loader, config, queue)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
