import os
import cv2
import sys
import torch as th
import numpy as np
import rerun as rr
from rerun import Box2DFormat
from data.event_rgb.event_rgb_dataset import EventRGBDataset
from modules.utils.detection import Mode, BackboneFeatureSelector
from modules.data.armasuisse import ArmaDataModule 
import hydra
from pathlib import Path
from utils.padding import InputPadderFromShape
from omegaconf import DictConfig, OmegaConf
from data.utils.types import DataType, LoaderDataDictGenX, DatasetMode
from modules.utils.fetch import fetch_data_module, fetch_model_module
from config.modifier import dynamically_modify_train_config
import torch.nn as nn
from tqdm import tqdm


def convert_image(img):
    img = img.clone()
    img = img.squeeze(0).permute(1,2,0).numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img * 255
    img = img.astype(np.uint8)
    return img

def convert_event(ev):
    ev = ev.clone()
    ev = torch.sum(ev, 0)
    heatmap = cv2.applyColorMap(sp_mask, cv2.COLORMAP_JET)


def logging(rr, start_time, events, images, labels):
    assert len(images) == len(events) and len(labels) == len(events)
    time = start_time
    for img, ev, ll in zip(images, events, labels):
        boxes = ll.object_labels[:, 1:5]
        tmp_img = convert_image(img)
        rr.set_time("stable_time", duration=time)
        rr.log("Images", rr.Image(tmp_img, color_model='BGR'))
        rr.log("Events", rr.Image(ev, color_model='BGR'))
        rr.log("Boxes", rr.Boxes2D(array=boxes, array_format=Box2DFormat.XYWH, colors=0x43ff64ff))


@hydra.main(config_path='config', config_name='train', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=False)
    sequence_length = config.dataset.sequence_length
    batch_size = config.batch_size.train

    dataset = EventRGBDataset
    dataloader = ArmaDataModule(config.dataset, 4, 4, batch_size, batch_size, dataset)
    dataloader.setup('fit')
    train_loader = dataloader.train_dataloader()

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = get_model(config)
    model.to(device)


    # thresholds = [0.05, 0.07, 0.09, 0.1, 0.12, 0.13, 0.2, 0.5, 0.8]
    batch_count = 0
    max_batch_count = 10
    time = 0
    count_img = 0

    for threshold in thresholds:
        train_loader = dataloader.train_dataloader()
        rr.init(f"threshold: {random}")
        rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")
        time = 0
        all_boxes = 0
        all_included = 0
        count = 0
        for batch in tqdm(train_loader):
            data = batch['data']
            events = data.get(DataType.EV_REPR)
            image = data.get(DataType.IMAGE)
            boxes = data.get(DataType.OBJLABELS_SEQ)
            boxes = extract_boxes(boxes, sequence_length)

            image = th.cat(image)
            fpn_output = model.mdl.fpn(backbone_features)
            sparsity_mask = get_sparsity_mask(fpn_output[0])
            sparsity_mask = sparsity_mask > threshold
            all, included, time = calculate_included(sparsity_mask, boxes, rr, time, image)
            all_boxes += all
            all_included += included

        print(f"Threshold: {threshold}: All_boxes: {all_boxes} Included_boxes: {all_included}")


if __name__ == '__main__':
    # th.multiprocessing.set_start_method('spawn')
    main()
