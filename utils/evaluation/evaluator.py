import os
import contextlib
from typing import Any, List, Optional, Dict
from warnings import warn

import numpy as np

from pycocotools.coco import COCO
from utils.evaluation.prophesee.evaluation import evaluate_list
try:
    coco_eval_type = 'cpp-based'
    from detectron2.evaluation.fast_eval_api import COCOeval_opt as COCOeval
except ImportError:
    coco_eval_type = 'python-based'
    from pycocotools.cocoeval import COCOeval


def convert_to_xywh(boxes):                                          
    xmin, ymin, xmax, ymax = boxes                         
    return [xmin, ymin, xmax-xmin, ymax-ymin]

class Evaluator:
    LABELS = 'lables'
    PREDICTIONS = 'predictions'

    def __init__(self, classes, height, width):
        self._buffer = None
        self._buffer_empty = True
        self._reset_buffer()
        self.classes = classes
        self.height = height
        self.width = width

    def _reset_buffer(self):
        self._buffer_empty = True
        self._buffer = {
            self.LABELS: list(),
            self.PREDICTIONS: list(),
        }

    def _add_to_buffer(self, key: str, value: List[np.ndarray]):
        assert isinstance(value, list)
        # for entry in value:
        #     assert isinstance(entry, np.ndarray, None), f"{entry=}"
        self._buffer_empty = False
        assert self._buffer is not None
        self._buffer[key].extend(value)

    def _get_from_buffer(self, key: str) -> List[np.ndarray]:
        assert not self._buffer_empty
        assert self._buffer is not None
        return self._buffer[key]

    def add_predictions(self, predictions: List[np.ndarray]):
        self._add_to_buffer(self.PREDICTIONS, predictions)

    def add_labels(self, labels: List[np.ndarray]):
        self._add_to_buffer(self.LABELS, labels)

    def reset_buffer(self) -> None:
        # E.g. call in on_validation_epoch_start
        self._reset_buffer()

    def has_data(self):
        return not self._buffer_empty

    def format_labels_to_coco(self, gt, im_id, id):
        annotations = []
        for bbox in gt:
            box = convert_to_xywh(bbox[:4])
            w, h = box[2:]
            area = w * h
            id += 1
            annotation = {
                "area": float(area),
                "iscrowd": False,
                "image_id": im_id,
                "bbox": box,
                "category_id": int(bbox[4]),
                "id": id
            }
            annotations.append(annotation)
        return annotations, id

    def format_predictions_to_coco(self, predictons, im_id):
        results = []
        if predictons is not None:
            for bbox in predictons:
                pred_box = convert_to_xywh(bbox[:4])
                image_result = {
                    'image_id': im_id,
                    'score': bbox[5],
                    'category_id': int(bbox[6]),
                    'bbox': pred_box,
                }
                # print(f"{image_result=}")
                results.append(image_result)
        
        return results

    def to_coco_format(self, gts, detections):
        """
        utilitary function producing our data in a COCO usable format
        """
        annotations = []
        results = []
        images = []
        box_id_count = 0

        categories = [{"id": id, "name": class_name, "supercategory": "none"}
                      for id, class_name in enumerate(self.classes)]
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
                 "height": self.height,
                 "width": self.width})

            annos, box_id_count = self.format_labels_to_coco(gt, im_id, box_id_count)
            annotations.extend(annos)

            preds = self.format_predictions_to_coco(pred, im_id)
            results.extend(preds)

        dataset = {"info": {},
                   "licenses": [],
                   "type": 'instances',
                   "images": images,
                   "annotations": annotations,
                   "categories": categories}
        return dataset, results

    def coco_eval(self, dataset, results, num_imgs):
        out_keys = ('AP (all)', 'AP (IoU=0.50)', 'AP (IoU=0.75)', 'AP (small)', 'AP (medium)',
                    'AP (large)', 'AR (all)', 'AR (small)', 'AR (medium)', 'AR (large)')
        out_dict = {k: 0.0 for k in out_keys}

        if len(results) == 0:
            # Corner case no predictions
            print('no detections for evaluation found.')
            return out_dict

        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            coco_gt = COCO()
            coco_gt.dataset = dataset
            coco_gt.createIndex()
            coco_pred = coco_gt.loadRes(results)

            coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
            coco_eval.params.imgIds = np.arange(1, num_imgs + 1, dtype=int)

            coco_eval.evaluate()
            coco_eval.accumulate()
            # info: https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
            coco_eval.summarize()
        for idx, key in enumerate(out_keys):
            out_dict[key] = coco_eval.stats[idx]
        # out_dict = coco_eval.stats.tolist()
        return out_dict

    def evaluate_buffer(self) -> Optional[Dict[str, Any]]:
        # e.g call in on_validation_epoch_end
        if self._buffer_empty:
            warn("Attempt to use prophesee evaluation buffer, but it is empty", UserWarning, stacklevel=2)
            return

        labels = self._get_from_buffer(self.LABELS)
        predictions = self._get_from_buffer(self.PREDICTIONS)
        assert len(labels) == len(predictions)
        dataset, results = self.to_coco_format(labels, predictions)
        # for anno in dataset['annotations']:
        #     print(f"{anno['id']=} {anno['bbox']=}")
        # for result in results:
        #     print(f"{result['bbox']=}")
        metrics = self.coco_eval(dataset, results, len(labels))
        return metrics
