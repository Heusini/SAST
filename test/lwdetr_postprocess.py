import cv2
import numpy as np
import torch as th
from testdata import TestData

import sys
sys.path.append(".")
from utils.evaluation.evaluator import Evaluator

from models.detection.lwdetr.lwdetr import PostProcess
from data.utils.object_labels import ObjectLabels
from data.arma_utils.labels import ObjectLabelFactory

def get_config_for_callbacks():
    class DictObj:
        def __init__(self, in_dict:dict):
            assert isinstance(in_dict, dict)
            for key, val in in_dict.items():
                if isinstance(val, (list, tuple)):
                   setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
                else:
                   setattr(self, key, DictObj(val) if isinstance(val, dict) else val)
    config = {}
    config['dataset'] = {}
    config['dataset']['name'] = "arma"
    config['dataset']['classes'] = ["drone"]
    config['logging'] = None
    config = DictObj(config)
    return config

def run_postprocess_tests():
    test_data = TestData()

    test_objlabels_to_processor(test_data)

def bounding_box_from_label(label):
    bbox = np.zeros((1, 5))
    bbox[:, 0] = label['x']
    bbox[:, 1] = label['y']
    bbox[:, 2] = label['w']
    bbox[:, 3] = label['h']
    bbox[:, 4] = label['class_id']
    box = bbox.copy()
    return box


def test_objlabels_to_processor(test_data):
    height = 384
    width = 640

    from callbacks.detection import DetectionVizCallback
    config = get_config_for_callbacks()
    detection = DetectionVizCallback(config)

    img = test_data.img
    label = test_data.label
    print(f"{label=}")
    bbox = bounding_box_from_label(label)
    bbox[:, 2:4] += bbox[:, :2]

    label = ObjectLabelFactory.from_structured_array(label,
                                                     (384, 640),
                                                     None)

    label = label.get_object_labels()

    lwdetr_labels = ObjectLabels.get_labels_as_batched_tensor([label], format_= "lwdetr")
    print(lwdetr_labels)
    boxes = lwdetr_labels[0]['boxes']
    scale_fct = th.tensor([width, height, width, height]).to(boxes.device)
    boxes =  boxes * scale_fct
    print(f"{boxes=}")
    boxes[:, 2:] += boxes[:, :2]

    img = detection.draw_bboxes(img, bbox, color=(0,0,255))
    img = detection.draw_bboxes(img, boxes.numpy(), color=(255,0,0))
    cv2.imshow("window", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_postprocess_tests()

