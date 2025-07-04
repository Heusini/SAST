import os
import cv2
import sys
import hydra
import numpy as np
sys.path.append(".")
from event_rgb_dataset import EventRGBDataset 
from data.general.partial_dataset import PartialDataset
from data.general.augmented import AugmentedDataset
from omegaconf import DictConfig, OmegaConf

from data.utils.types import DataType, LoaderDataDictGenX, DatasetMode
import cv2
import numpy as np
from typing import List
import bbox_visualizer as bbv

def extract_bounding_boxes(labels: np.ndarray) -> np.ndarray:
    # stacked = np.column_stack([labels[field] for field in labels.dtype.names])
    new_bbs = labels[:, 1:5]
    if new_bbs.ndim == 1:
        new_bbs = np.expand_dims(new_bbs, axis=0)
    new_bbs[:, 2:] += new_bbs[:, :2]
    # new_bbs = new_bbs.astype(np.int32)
    return new_bbs

def draw_rgb_and_bbox(img_data: np.ndarray, bboxes: np.ndarray):
    img = img_data
    if bboxes is not None:
        img = bbv.draw_multiple_rectangles(img, bboxes.tolist(), thickness=1)
    cv2.imshow('window', img)

def draw_and_display(event_data: np.ndarray,
                     rgb_data: np.ndarray,
                     new_bb: np.ndarray,
                     window_name: str ='window'):
    event = event_data
    rgb = rgb_data
    if len(event_data.shape) == 4:
        event = np.sum(event, axis=0)
    event = np.sum(event, axis=0)

    event = event / np.max(event)
    event = event * 255
    event = np.array(event, np.uint8)
    event = cv2.applyColorMap(event, cv2.COLORMAP_VIRIDIS)
    if new_bb is not None and len(new_bb) > 0:
        event = bbv.draw_multiple_rectangles(event, new_bb.tolist(), thickness=1)
        rgb = bbv.draw_multiple_rectangles(rgb, new_bb.tolist(), thickness=1)

    combined_image = np.hstack((event, rgb))
    
    cv2.imshow(window_name, combined_image)

def draw_and_wait(event_data, rgb_data, new_bb):
    draw_and_display(event_data, rgb_data, new_bb)
    if cv2.waitKey(0) == ord("q"):
        cv2.destroyAllWindows()
        sys.exit(0)



@hydra.main(config_path='../../config', config_name='train', version_base='1.2')
def main(config: DictConfig):
    event_dataset = EventRGBDataset.build(DatasetMode.TRAIN, config.dataset)
    partial = PartialDataset(event_dataset, 0.2, True)
    augmented = AugmentedDataset.build(config.dataset, partial)
    for data in augmented:
        sequence_len = len(data[DataType.IMAGE])
        for i in range(sequence_len):
            bboxes = extract_bounding_boxes(data[DataType.OBJLABELS_SEQ][i].object_labels.numpy()).astype(np.int32)
            events = data[DataType.EV_REPR][i].numpy()
            rgbs = data[DataType.IMAGE][i]
            rgbs = rgbs.permute(1, -1, 0).numpy()
            draw_and_wait(events, rgbs, bboxes)

if __name__ == "__main__":
    main()



