import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path

import torch
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

import matplotlib.pyplot as plt

def draw_plot(image, mask):
    img = image
    if len(image.shape) == 4:
        img = np.sum(img, axis=0)
    img = np.sum(img, axis=0)
    new_mask = np.concatenate(mask)
    new_mask = new_mask.reshape((384, 640))
    mask_normalized = (new_mask-new_mask.min()) / (new_mask.max() - new_mask.min())
    print(mask_normalized)

    plt.imshow(img)
    # plt.imshow(np.zeros_like(img), alpha=0)
    # plt.imshow(np.dstack((np.ones_like(mask_normalized),
    #                       np.zeros_like(mask_normalized),
    #                       np.zeros_like(mask_normalized),
    #                       )),
    #            alpha=mask_normalized)
    plt.show()


def xy_from_index(index, height, width):
    y = index // width
    x = index - (width * y)

    return x, y

HEIGHT = 384
WIDTH = 640

def draw_and_display(img_data, masks, bboxes, window_name='window'):
    height, width = masks.shape
    size = HEIGHT // masks.shape[0]
    img = img_data
    if len(img_data.shape) == 4:
        img = np.sum(img, axis=0)
    img = np.sum(img, axis=0)

    # img = np.sum(img, axis=0)
    # print(img.shape)
    img = img / np.max(img)
    img = img * 255
    img = np.array(img, np.uint8)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    # img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    mask_color = np.zeros((size, size, 3), dtype=np.uint8)
    mask_color[:, :, 2] = 255

    if bboxes is not None:
        new_bb = bboxes
        new_bb[:, 2:] += new_bb[:, :2]
        new_bb = new_bb.astype(np.int32)
        img = bbv.draw_multiple_rectangles(img, new_bb.tolist(), thickness=1)

    for y_i in range(height):
        for x_i in range(width):
            alpha = masks[y_i][x_i]
            x = x_i * size
            y = y_i * size
            img[y:y+size, x:x+size] = cv2.addWeighted(mask_color, alpha, img[y:y+size, x:x+size], 1-alpha, 0)

    y_repeat = int(np.ceil(HEIGHT/ height))
    x_repeat = int(np.ceil(WIDTH / width))

    heatmap = np.repeat(np.repeat(masks * 255, y_repeat, axis=0), x_repeat, axis=1)
    heatmap = heatmap.astype(np.uint8)
    print(heatmap.shape)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
    print(height, width)
    out_img = np.hstack([img, heatmap])
    cv2.imshow(window_name, out_img)

def draw_and_wait(img_data, masks = None, bboxes = None):
    draw_and_display(img_data, masks, bboxes)
    if cv2.waitKey(0) == ord("q"):
        cv2.destroyAllWindows()
        sys.exit(0)

def map_tokens_to_image(frame, tokens, bboxes):
    print(frame.shape)
    sf = 0
    print(f"{tokens[sf].shape=}")
    # current_token = np.sum(tokens[sf][0].numpy(), axis=0)

    current_token = torch.norm(tokens[sf][0], dim=0)
    print(current_token.shape)
    min_val = current_token.min()
    max_val = current_token.max()
    normalized_scores = (current_token - min_val) / (max_val - min_val + 1e-8)
    draw_and_display(frame, normalized_scores.numpy(), bboxes)
    if cv2.waitKey(500) == ord("q"):
        cv2.destroyAllWindows()
        sys.exit(0)
    # draw_plot(frame, masks)

@hydra.main(config_path='config', config_name='val', version_base='1.2')
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
    module = module.load_from_checkpoint(str(ckpt_path), **{'full_config': config}, strict=True)

    module.eval()
    # Get a batch (or a single sample wrapped as batch)
    data_module.setup('validate')
    val_loader = data_module.val_dataloader()

    in_res_hw = tuple(config.model.backbone.in_res_hw)
    input_padder = InputPadderFromShape(desired_hw=in_res_hw)
    # batch = next(iter(val_loader))  # or your own custom input
    # No gradient computation needed during inference
    for batch in val_loader:
        data = batch['data']
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                ev_tensor_sequence = data[DataType.EV_REPR]
                sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
                is_first_sample = data[DataType.IS_FIRST_SAMPLE]
                token_mask_sequence = data.get(DataType.TOKEN_MASK, None)
                sequence_len = len(ev_tensor_sequence)
                batch_size = ev_tensor_sequence[0].shape[0]
                for tidx in range(sequence_len):
                    ev_tensors = ev_tensor_sequence[tidx]
                    ev_tensors = input_padder.pad_tensor_ev_repr(ev_tensors)
                    bboxes = sparse_obj_labels[tidx][0]
                    new_bbs = None
                    if bboxes:
                        new_bb = bboxes.object_labels[:, 1:5]
                        new_bbs = np.vstack(new_bb)

                    preds = module.predict_step(ev_tensors, batch_idx=0)
                    map_tokens_to_image(ev_tensors.numpy(), preds, new_bbs)



if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
