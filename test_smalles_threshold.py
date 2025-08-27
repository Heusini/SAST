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


PATCH_SIZE = 16
LOGGING = False

def get_model(config):
    module = fetch_model_module(config=config)
    ckpt_path = Path(config.checkpoint)
    ckpt = th.load(ckpt_path, map_location='cpu')
    state_dict = ckpt["state_dict"]

    backbone_str = "mdl.backbone."
    backbone_dict = {k.replace(backbone_str, ""): v 
                 for k, v in state_dict.items() if k.startswith(backbone_str)}

    fpn_str = "mdl.fpn."
    fpn_dict = {k.replace(fpn_str, ""): v 
                 for k, v in state_dict.items() if k.startswith(fpn_str)}

    module.mdl.backbone.load_state_dict(backbone_dict, strict=True)
    module.mdl.fpn.load_state_dict(fpn_dict, strict=True)
    for param in module.mdl.backbone.parameters():
        param.requires_grad = False

    for param in module.mdl.fpn.parameters():
        param.requires_grad = False

    module.mdl.backbone.eval()
    module.mdl.fpn.eval()
    return module

def forward_sast_backbone(model, ev_tensor_sequence, sequence_len):
    backbone_feature_selector = BackboneFeatureSelector()
    input_padder = InputPadderFromShape(desired_hw=(384, 640))

    prev_states = None
    for tidx in range(sequence_len):
        ev_tensors = ev_tensor_sequence[tidx]
        ev_tensors = ev_tensors.to(dtype=model.dtype).to(device=model.device)
        ev_tensors = input_padder.pad_tensor_ev_repr(ev_tensors)

        backbone_features, states, _ = model.mdl.forward_backbone(x=ev_tensors,
                                                              previous_states=prev_states)
        prev_states = states

        # current_labels = [l for l in sparse_obj_labels[tidx].sparse_object_labels_batch]
        # obj_labels.extend(current_labels)
        # event_repr.extend(x[0] for x in ev_tensors.split(1))
        backbone_feature_selector.add_backbone_features(backbone_features)
    return backbone_feature_selector.get_batched_backbone_features()

def get_sparsity_mask(fpn_layer: th.Tensor):
    max_pool = nn.MaxPool2d(2, 2)
    tokens = fpn_layer
    tokens = th.norm(tokens, dim=1)
    min_val = tokens.amin(dim=(-2, -1), keepdim=True)
    max_val = tokens.amax(dim=(-2, -1), keepdim=True)

    tokens = (tokens - min_val) / (max_val-min_val + 1e-8)
    tokens = max_pool(tokens)

    return tokens

def is_fully_included(box_xywh, patch_mask) -> bool:
    return is_percent_included(box_xywh, patch_mask, 1)

def is_percent_included(box_xywh, patch_mask, percent) -> bool:
    box_xyxy = box_xywh.clone()
    box_xyxy[2:] += box_xyxy[:2]
    x_min = int(box_xyxy[0])
    x_max = int(np.ceil(box_xyxy[2]))
    y_min = int(box_xyxy[1])
    y_max = int(np.ceil(box_xyxy[3]))
    pt_mask = th.kron(patch_mask, th.ones((16, 16), device=patch_mask.device)).int()
    mask = th.zeros(pt_mask.shape, device=patch_mask.device).int()
    mask[y_min:y_max+1, x_min:x_max+1] = 1

    fully_included = th.bitwise_and(pt_mask, mask).sum()
    return fully_included >= box_xywh[2] * box_xywh[3] * percent

def extract_boxes(boxes, sequence_length):
    new_boxes = []
    for i in range(sequence_length):
        tmp = [b for b in boxes[i].sparse_object_labels_batch]
        new_boxes.extend(tmp)
    return new_boxes

def sparsity_to_image(sparsity_mask): 
    sp_mask = sparsity_mask.cpu().numpy().astype(np.uint8) * 255
    sp_mask = np.kron(sp_mask, np.ones((16, 16), dtype=np.uint8))
    heatmap = cv2.applyColorMap(sp_mask, cv2.COLORMAP_JET)
    return heatmap


def convert_image(img):
    img = img.clone()
    img = img.squeeze(0).permute(1,2,0).numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img * 255
    img = img.astype(np.uint8)
    return img

def calculate_included(sparsity_masks, labels, rr, start_time, image):
    time = start_time
    all_boxes = 0
    included_boxes = 0
    assert len(sparsity_masks) == len(labels) and len(labels) == len(image)
    for sp, ll, img in zip(sparsity_masks, labels, image):
        boxes = ll.object_labels[:, 1:5]
        not_included = []
        included = []
        for i, bb in enumerate(boxes):
            all_boxes += 1
            if is_percent_included(bb, sp, 0.5):
                included_boxes += 1
                included.append(i)
            else:
                not_included.append(i)
        if LOGGING and len(not_included) > 0: 
            rr.set_time("stable_time", duration=time)
            heatmap = sparsity_to_image(sp)
            rr.log("mask", rr.Image(heatmap, color_model='BGR')) 
            tmp_img = convert_image(img)
            rr.log("IMAGE", rr.Image(tmp_img, color_model='BGR'))
            rr.log("BOXES_EXCLUDED", rr.Boxes2D(array=boxes[not_included], array_format=Box2DFormat.XYWH, colors=0x43ff64ff))
            rr.log("BOXES_INCLUDED", rr.Boxes2D(array=boxes[included], array_format=Box2DFormat.XYWH))

            time+=1
    return all_boxes, included_boxes, time


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


    thresholds = [0.05, 0.07, 0.09, 0.1, 0.12, 0.13, 0.2, 0.5, 0.8]
    batch_count = 0
    max_batch_count = 10
    time = 0
    count_img = 0

    for threshold in thresholds:
        train_loader = dataloader.train_dataloader()
        if LOGGING:
            rr.init(f"threshold: {threshold}")
            rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")
        time = 0
        all_boxes = 0
        all_included = 0
        count = 0
        for batch in tqdm(train_loader):
            with th.autocast(device_type='cuda', dtype=th.float16):
                data = batch['data']
                events = data.get(DataType.EV_REPR)
                image = data.get(DataType.IMAGE)
                boxes = data.get(DataType.OBJLABELS_SEQ)
                boxes = extract_boxes(boxes, sequence_length)

                image = th.cat(image)
                backbone_features = forward_sast_backbone(model, events, sequence_length)
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
