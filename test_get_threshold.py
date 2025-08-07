import os

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

import matplotlib.pyplot as plt

HEIGHT = 384
WIDTH = 640
def apply_sparsity_mask(img, sparsity_mask, alpha = 0.5): 
    # a lot of magic numbers -> change somehow
    patch_size = int(np.sqrt((img.shape[0] * img.shape[1])/sparsity_mask.shape[0]))
    sp_mask = sparsity_mask.reshape(img.shape[0]//patch_size, img.shape[1]//patch_size)
    print(f"{sp_mask.shape=}")
    sp_mask = np.kron(sp_mask, np.ones((patch_size, patch_size), dtype=sparsity_mask.dtype))
    print(f"{sp_mask.shape=}")
    mask_img = np.zeros_like(img)
    # mask_img[:, :, 3][sp_mask] = 255
    mask_img[:, :, 2][sp_mask] = 255
    img = cv2.addWeighted(mask_img, alpha, img, 1-alpha, 0)

    return img

def draw_boxes(img, bboxes, color = (255, 255, 255)):
    for box in bboxes:
        new_bb = box.numpy().copy()
        new_bb[2:] += new_bb[:2]
        new_bb = new_bb.astype(np.int32)
        img = bbv.draw_rectangle(img, new_bb, thickness=1, bbox_color=color)

    return img

def draw_and_display(img_data, tokens, mask, bboxes,image = None,predictions =None, window_name='window'):
    height, width = tokens.shape
    print(height, width)
    img = img_data
    if len(img_data.shape) == 4:
        img = np.sum(img, axis=0)
    img = np.sum(img, axis=0)

    img = cv2.applyColorMap(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)

    img = apply_sparsity_mask(img, mask)

    if predictions is not None:
        new_pred = []
        for p in predictions:
            if p is None:
                continue
            pred = p.numpy()
            pred = pred[:, :4]
        # new_pred[:, 2:] += new_pred[:, :2]
            pred = new_bb.astype(np.int32)
            if len(pred) > 1:
                pred = pred.tolist()
            new_pred.extend(pred)
        print(new_pred)
        if len(new_pred) > 0:
            img = bbv.draw_multiple_rectangles(img, new_pred, bbox_color=(0,0,255), thickness=1)

    y_repeat = int(np.ceil(HEIGHT/ height))
    x_repeat = int(np.ceil(WIDTH / width))


    heatmap = np.zeros((HEIGHT, WIDTH))
    heatmap = np.repeat(np.repeat(tokens, y_repeat, axis=0), x_repeat, axis=1) * 255
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.applyColorMap(cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)


    img = draw_boxes(img, bboxes)
    heatmap = draw_boxes(heatmap, bboxes)

    out_img = np.hstack([img, heatmap])
    if image is not None:
        image = image.squeeze(0).permute(1,2,0).numpy()
        # image = cv2.cvtColor(image, cv2.COLOR_)
        image = image * 255
        image = image.astype(np.uint8)
        print(f"{image.shape=}")
        print(f"{out_img.shape=}")
        if bboxes is not None:
            image = draw_boxes(image, bboxes)
        if predictions is not None:
            if len(new_pred) > 0:
                image = bbv.draw_multiple_rectangles(image, new_pred, thickness=1, bbox_color=(0, 0, 255))

        out_img = np.hstack([out_img, image])
    cv2.imshow(window_name, out_img)

def map_tokens_to_image(frame, tokens, mask, bboxes, image=None, predictions=None):
    # current_token = np.sum(tokens[sf][0].numpy(), axis=0)
    draw_and_display(frame, tokens.numpy(), mask, bboxes, image, predictions)
    if cv2.waitKey(0) == ord("q"):
        cv2.destroyAllWindows()
        sys.exit(0)

@hydra.main(config_path='config', config_name='train', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    # print('------ Configuration ------')
    # print(OmegaConf.to_yaml(config))
    # print('---------------------------')

    gpus = config.hardware.gpus
    assert isinstance(gpus, int), 'no more than 1 GPU supported'
    gpus = [gpus]

    # ---------------------
    # Data
    # ---------------------
    data_module = fetch_data_module(config=config)

    # ---------------------
    # Logging and Checkpoints
    # ---------------------
    logger = CSVLogger(save_dir='./validation_logs')
    ckpt_path = Path(config.checkpoint)

    # ---------------------
    # Model
    # ---------------------
    
    module = fetch_model_module(config=config)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt["state_dict"]
    backbone_dict = {k.replace("mdl.backbone.", ""): v 
                 for k, v in state_dict.items() if k.startswith("mdl.backbone.")}
    
    fpn_dict = {k.replace("mdl.fpn.", ""): v 
                 for k, v in state_dict.items() if k.startswith("mdl.fpn.")}

    module.mdl.backbone.load_state_dict(backbone_dict, strict=True)
    module.mdl.fpn.load_state_dict(fpn_dict, strict=True)
    for param in module.mdl.backbone.parameters():
        param.requires_grad = False

    for param in module.mdl.fpn.parameters():
        param.requires_grad = False

    module.eval()

    # Get a batch (or a single sample wrapped as batch)
    data_module.setup('fit')
    val_loader = data_module.train_dataloader()

    in_res_hw = tuple(config.model.backbone.in_res_hw)
    input_padder = InputPadderFromShape(desired_hw=in_res_hw)
    # batch = next(iter(val_loader))  # or your own custom input
    # No gradient computation needed during inference
    mdl_config = config.model

    max_pool = nn.MaxPool2d(2,2)
    for batch in val_loader:
        data = batch['data']
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                ev_tensor_sequence = data[DataType.EV_REPR]
                sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
                is_first_sample = data[DataType.IS_FIRST_SAMPLE]
                image = data.get(DataType.IMAGE)
                token_mask_sequence = data.get(DataType.TOKEN_MASK, None)
                sequence_len = len(ev_tensor_sequence)
                batch_size = ev_tensor_sequence[0].shape[0]
                print(f"{batch_size=}")
                for tidx in range(sequence_len):
                    ev_tensors = ev_tensor_sequence[tidx]
                    ev_tensors = input_padder.pad_tensor_ev_repr(ev_tensors)
                    bboxes = sparse_obj_labels[tidx]
                    bb_list = []
                    for box in bboxes:
                        new_bb = box.object_labels[:, 1:5]
                        new_bbs = []
                        for bb in new_bb:
                            new_bbs.append(bb)
                        bb_list.append(new_bbs)

                    preds, _, _ = module.mdl.backbone(ev_tensors)
                    # for i in range(len(preds)):
                    #     print(preds[i].shape)
                    fpn_layers = module.mdl.fpn(preds)
                    for layer in fpn_layers:
                        print(f"{layer.shape=}")

                    tokens = module.mdl.get_sparsity_mask(fpn_layers[1])
                    print(tokens.shape)
                    sparsity_mask = tokens > 0.15
                    # max = np.max((sparsity_mask.shape[-1], sparsity_mask.shape[-2]))
                    # sparsity_mask, pad = InputPadderFromShape._pad_tensor_impl(sparsity_mask, (max, max), mode='constant', value=False)
                    sparsity_mask = sparsity_mask.flatten(1,2)
                    # sparsity_mask = np.ones_like(sparsity_mask)
                    sparsity_mask = sparsity_mask.numpy()
                    print(f"{sparsity_mask.shape=}")

                    # for i in range(len(preds)):
                    #     preds[i] = max_pool(preds[i])
                    # preds = [preds[i] for i in [1, 2, 3, 4]]
                    img = None
                    if image is not None:
                        img = image[tidx]
                        img = input_padder.pad_tensor_ev_repr(img)
                        print(f"{img.shape=}")
                    predictions = None
                    print(f"{bb_list=}")
                    for i in range(batch_size):
                        map_tokens_to_image(ev_tensors[i].numpy(), tokens[i], sparsity_mask[i], bb_list[i], img[i], predictions)

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
