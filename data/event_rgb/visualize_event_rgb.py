import os
import cv2
import sys
import rerun as rr
from rerun import Box2DFormat
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
from tqdm import tqdm

def extract_bounding_boxes(labels: np.ndarray) -> np.ndarray:
    # stacked = np.column_stack([labels[field] for field in labels.dtype.names])
    new_bbs = labels[:, 1:5]
    # if new_bbs.ndim == 1:
    #     new_bbs = np.expand_dims(new_bbs, axis=0)
    # new_bbs[:, 2:] += new_bbs[:, :2]
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
    event = cv2.applyColorMap(event, cv2.COLORMAP_JET)
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

def convert_image(img):
    img = img.clone()
    img = img.squeeze(0).permute(1,2,0).numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img * 255
    img = img.astype(np.uint8)
    return img

def event_to_image(event): 
    event = np.sum(event, axis=0)
    event = cv2.applyColorMap(cv2.normalize(event, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
    return event

@hydra.main(config_path='../../config', config_name='train', version_base='1.2')
def main(config: DictConfig):
    OmegaConf.to_container(config, resolve=True, throw_on_missing=False)
    event_dataset = EventRGBDataset.build(DatasetMode.TRAIN, config.dataset)
    partial = PartialDataset(event_dataset, 0.2, True)
    augmented = AugmentedDataset.build(config.dataset, partial)

    skip = True
    previous_image = None
    previous_event = None
    previous_boxes = None
    rr.init(f"nerd_events")
    rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")
    time = 0
    empty_again = False
    count = 0
    max_count = 1000
    skip_num = 1000
    for data in tqdm(event_dataset):
        sequence_len = len(data[DataType.IMAGE])
        for i in range(sequence_len):
            bboxes = extract_bounding_boxes(data[DataType.OBJLABELS_SEQ][i].object_labels.numpy()).astype(np.int32)
            if skip and len(bboxes) == 0:
                continue
            else:
                skip = False

            if skip_num > count:
                count += 1
                continue
            events = data[DataType.EV_REPR][i].numpy()
            # path = data[DataType.EVENT_PATH]
            events = event_to_image(events)
            rgbs = data[DataType.IMAGE][i]
            rgbs = convert_image(rgbs)

            rr.set_time("stable_time", duration=time)
            rr.log("IMAGE", rr.Image(rgbs, color_model='RGB'))
            rr.log("event", rr.Image(events, color_model='BGR'))
            rr.log("boxes", rr.Boxes2D(array=bboxes, array_format=Box2DFormat.XYWH))
            time += 0.033
            # if not empty_again and len(bboxes) == 0:
            #     print(path)
            #     rr.set_time("stable_time", duration=time)
            #     rr.log("IMAGE", rr.Image(previous_image, color_model='BGR'))
            #     rr.log("event", rr.Image(previous_event, color_model='BGR'))
            #     rr.log("boxes", rr.Boxes2D(array=previous_boxes, array_format=Box2DFormat.XYWH))
            #     time += 1
            #     rr.set_time("stable_time", duration=time)
            #     rr.log("IMAGE", rr.Image(rgbs, color_model='BGR'))
            #     rr.log("event", rr.Image(events, color_model='BGR'))
            #     rr.log("boxes", rr.Boxes2D(array=bboxes, array_format=Box2DFormat.XYWH))
            #     empty_again = True
            # if empty_again:
            #     empty_again = False
            #     time += 1
            #     rr.set_time("stable_time", duration=time)
            #     rr.log("IMAGE", rr.Image(rgbs, color_model='BGR'))
            #     rr.log("event", rr.Image(events, color_model='BGR'))
            #     rr.log("boxes", rr.Boxes2D(array=bboxes, array_format=Box2DFormat.XYWH))
            #     time += 10

            # previous_image = rgbs
            # previous_event = events
            # previous_boxes = bboxes
            count += 1
        if count > max_count+skip_num:
            break



if __name__ == "__main__":
    main()



