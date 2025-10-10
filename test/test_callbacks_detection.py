import cv2
import torch
import numpy as np
from testdata import TestData

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

def test_callbacks_detection():
    import sys
    sys.path.append(".")
    from data.utils.types import ObjDetOutput
    from callbacks.detection import DetectionVizCallback

    test_data = TestData()
    config = get_config_for_callbacks()
    detection = DetectionVizCallback(config)
    test_empty_labels(test_data, detection)
    test_empty_tensor(test_data, detection)
    test_callbacks_detection_drawing(test_data, detection)

def test_empty_labels(test_data, detection):
    img = test_data.img
    label = [None]
    detection.draw_bboxes(img, label, color=(0,0,255))

def test_empty_tensor(test_data, detection):
    img = test_data.img
    label = torch.empty((0, 5))
    label = label.numpy()
    detection.draw_bboxes(img, label, color=(0,0,255))


def test_callbacks_detection_drawing(test_data, detection):
    img = test_data.img
    label = test_data.label
    bbox = np.zeros((1, 5))
    bbox[:, 0] = label['x']
    bbox[:, 1] = label['y']
    bbox[:, 2] = label['x'] + label['w']
    bbox[:, 3] = label['y'] + label['h']
    bbox[:, 4] = label['class_id']
    box = bbox.copy()

    img = detection.draw_bboxes(img, box, color=(0, 0, 255))
    cv2.imshow("window", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_callbacks_detection()
