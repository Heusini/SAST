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
from einops import rearrange, reduce

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

def apply_sparsity_mask(img, sparsity_mask, alpha = 0.4): 
    # a lot of magic numbers -> change somehow
    sp_mask = sparsity_mask.reshape(24,40)
    sp_mask = np.kron(sp_mask, np.ones((16, 16), dtype=sparsity_mask.dtype))
    mask_img = np.zeros_like(img)
    # mask_img[:, :, 3][sp_mask] = 255
    mask_img[:, :, 0][sp_mask] = 255
    img = cv2.addWeighted(mask_img, alpha, img, 1-alpha, 0)
    return img

def draw_bboxes(image, bboxes, color):
    for box in bboxes:
        if box is None or len(box) == 0:
            continue
        bb = box.copy().astype(np.int32)
        bb = bb[:4]
        image = bbv.draw_rectangle(image, bb, bbox_color=color, thickness=1)
        # bbox_txt = self.get_bbox_text(box)
        # image = bbv.add_label(image, bbox_txt, bb, text_bg_color=color, size=0.2,top=True)
    return image

RED=(255, 0, 0)
BLUE=(0, 0, 255)
GREEN=(0, 255, 0)
def draw_and_wait(image):
    cv2.imshow("window", image)
    if cv2.waitKey(0) == ord("q"):
        cv2.destroyAllWindows()
        sys.exit(0)

def ev_repr_to_img(x: np.ndarray):
    ch, ht, wd = x.shape[-3:]
    assert ch > 1 and ch % 2 == 0
    ev_repr_reshaped = rearrange(x, '(posneg C) H W -> posneg C H W', posneg=2)
    img_neg = np.asarray(reduce(ev_repr_reshaped[0], 'C H W -> H W', 'sum'), dtype='int32')
    img_pos = np.asarray(reduce(ev_repr_reshaped[1], 'C H W -> H W', 'sum'), dtype='int32')
    img_diff = img_pos - img_neg
    img = 127 * np.ones((ht, wd, 3), dtype=np.uint8)
    img[img_diff > 0] = 255
    img[img_diff < 0] = 0
    return img

def get_bbox_from_label(label):
    bb = label.copy()
    bb = np.column_stack([bb[field] for field in label.dtype.names])
    bb = bb[:, 1:5]
    bb[:, 2:] += bb[:, :2]
    return bb

@hydra.main(config_path='config', config_name='train', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    # Just to check whether config can be resolved
    OmegaConf.to_container(config, resolve=True, throw_on_missing=False)

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
    print(f"{config.dataset.train.shuffle=}")

    # ---------------------
    # Logging and Checkpoints
    # ---------------------
    logger = CSVLogger(save_dir='./validation_logs')
    ckpt_path = Path(config.checkpoint)

    # ---------------------
    # Model
    # ---------------------
    
    module = fetch_model_module(config=config)
    # module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config}, strict=True)
    if ckpt_path:
        print('Resuming only the weights instead of the full training state')
        ckpt = torch.load(ckpt_path, map_location='cpu')
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
        ckpt_path = None

    module.eval()
    module.cuda()
    # Get a batch (or a single sample wrapped as batch)
    event_folder = Path("events")
    label_folder = Path("labels")
    frame_folder = Path("rgbs")
    # path = Path("/datasets/sheusinger/st_stephan_360_640_20/train/2024_01_10_115957_mavic_003/")
    # path = Path("/datasets/sheusinger/st_stephan_360_640_20/train/2024_01_10_170509_himo_011/")
    path = Path("/datasets/sheusinger/st_stephan_360_640_20/train/2024_01_10_170509_himo_002")
    # path = Path("/datasets/sheusinger/st_stephan_360_640_20/val/2024_01_10_170509_himo_007")
    event_path = path / event_folder
    label_path = path / label_folder
    rgb_path = path  / frame_folder

    event_files = os.listdir(event_path)
    label_files = os.listdir(label_path)
    rgb_files = os.listdir(rgb_path)
    event_files.sort(key=lambda item: (len(item), item))
    label_files.sort(key=lambda item: (len(item), item))
    rgb_files.sort(key=lambda item: (len(item), item))
    in_res_hw = tuple(config.model.backbone.in_res_hw)
    input_padder = InputPadderFromShape(desired_hw=in_res_hw)

    for ev_name, im_name, ll_name in zip(event_files, rgb_files, label_files):
        ev_tensor = np.load(event_path / ev_name)['arr_0']
        label = np.load(label_path / ll_name)['arr_0']

        ev_tensor = torch.from_numpy(ev_tensor).unsqueeze(0)
        frame = np.load(rgb_path / im_name)
        frame = frame[list(frame.keys())[0]]
        frame = torch.from_numpy(frame)
        frame = frame.permute(-1, 0, 1)
        frame = frame / 255.0


        print(f"{frame.shape=}")
        frame = input_padder.pad_tensor_ev_repr(frame)
        frame = frame.permute(1,2,0)
        frame = (frame * 255).numpy().astype(np.uint8)
        ev_tensor = input_padder.pad_tensor_ev_repr(ev_tensor)
        ev_tensor = ev_tensor.cuda()
        output, sparsity_mask, _, fpn_features  = module.mdl(ev_tensor, None)

        tokens = torch.norm(fpn_features[0][0], dim=0)
        min_val = tokens.min()
        max_val = tokens.max()
        normalized_scores = (tokens - min_val) / (max_val - min_val + 1e-8)
        masks = normalized_scores.cpu().numpy()


        gt_labels = get_bbox_from_label(label)

        image = ev_tensor.squeeze(0).cpu().numpy()
        image = ev_repr_to_img(image)
        image = apply_sparsity_mask(image, sparsity_mask.cpu().numpy(), alpha=0.4)
        image = draw_bboxes(image, gt_labels, GREEN)
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = draw_bboxes(frame, gt_labels, GREEN)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{config.checkpoint}.png", image)
        cv2.imwrite(f"rgb_frame.png", frame)

        image = np.hstack((image, frame))

        height, width = masks.shape
        y_repeat = int(np.ceil(384/ height))
        x_repeat = int(np.ceil(640 / width))
        heatmap = np.repeat(np.repeat(masks * 255, y_repeat, axis=0), x_repeat, axis=1)
        heatmap = heatmap.astype(np.uint8)
        heatmap = cv2.applyColorMap(cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(f"heatmap.png", heatmap)

        draw_and_wait(heatmap)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
