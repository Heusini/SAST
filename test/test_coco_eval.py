import numpy as np
from testdata import TestData

import sys
sys.path.append(".")
from utils.evaluation.evaluator import Evaluator

def run_evaluator_tests():
    test_data = TestData()

    test_label_eq_pred(test_data)
    print()
    test_none_predictions(test_data)
    print()
    test_empty_predictions(test_data)

def bounding_box_from_label(label):
    bbox = np.zeros((1, 5))
    bbox[:, 0] = label['x']
    bbox[:, 1] = label['y']
    bbox[:, 2] = label['x'] + label['w']
    bbox[:, 3] = label['y'] + label['h']
    bbox[:, 4] = label['class_id']
    box = bbox.copy()
    return box


def test_label_eq_pred(test_data):
    label = test_data.label
    box = bounding_box_from_label(label)

    prediction = np.ones((1,7))
    prediction[:, 6] = 0
    prediction[:, :4] = box[:, :4]
    evaluator = Evaluator(["drone"], 360, 640)
    evaluator.add_labels([box])
    evaluator.add_predictions([prediction])
    output = evaluator.evaluate_buffer()
    for key in output.keys():
        print(f"{key}: {output[key]:0.2f}")

def test_empty_predictions(test_data):
    label = test_data.label
    box = bounding_box_from_label(label)
    prediction = np.empty((0, 7))

    evaluator = Evaluator(["drone"], 360, 640)
    evaluator.add_labels([box])
    evaluator.add_predictions([prediction])
    output = evaluator.evaluate_buffer()
    for key in output.keys():
        print(f"{key}: {output[key]:0.2f}")

def test_none_predictions(test_data):
    label = test_data.label
    box = bounding_box_from_label(label)
    prediction = None

    evaluator = Evaluator(["drone"], 360, 640)
    evaluator.add_labels([box])
    evaluator.add_predictions([prediction])
    output = evaluator.evaluate_buffer()
    for key in output.keys():
        print(f"{key}: {output[key]:0.2f}")


if __name__ == "__main__":
    run_evaluator_tests()
