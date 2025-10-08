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

def draw_plot(image, mask):
    img = image
    if len(image.shape) == 4:
        img = np.sum(img, axis=0)
    img = np.sum(img, axis=0)
    new_mask = np.concatenate(mask)
    new_mask = new_mask.reshape((384, 640))
    mask_normalized = (new_mask-new_mask.min()) / (new_mask.max() - new_mask.min())

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

def draw_and_display(img_data, masks, bboxes,image = None,predictions =None, window_name='window'):
    height, width = masks.shape
    size = HEIGHT // masks.shape[0]
    img = img_data
    if len(img_data.shape) == 4:
        img = np.sum(img, axis=0)
    img = np.sum(img, axis=0)

    img = cv2.applyColorMap(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)

    mask_color = np.zeros((size, size, 3), dtype=np.uint8)
    mask_color[:, :, 2] = 255

    if bboxes is not None:
        new_bb = bboxes
        new_bb[:, 2:] += new_bb[:, :2]
        new_bb = new_bb.astype(np.int32)
        img = bbv.draw_multiple_rectangles(img, new_bb.tolist(), thickness=1)
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
            img = bbv.draw_multiple_rectangles(img, new_pred, bbox_color=(255,0,0), thickness=1)

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
    heatmap = cv2.applyColorMap(cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)

    # img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
    out_img = np.hstack([img, heatmap])
    if image is not None:
        print(f"{image.shape=}")
        image = image.squeeze(0).permute(1,2,0).numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image * 255
        image = image.astype(np.uint8)
        if bboxes is not None:
            image = bbv.draw_multiple_rectangles(image, new_bb.tolist(), thickness=1)
        if predictions is not None:
            if len(new_pred) > 0:
                image = bbv.draw_multiple_rectangles(image, new_pred, thickness=1, bbox_color=(255, 0, 0))

        out_img = np.hstack([out_img, image])
    cv2.imshow(window_name, out_img)

def draw_and_wait(img_data, masks = None, bboxes = None):
    draw_and_display(img_data, masks, bboxes)
    if cv2.waitKey(0) == ord("q"):
        cv2.destroyAllWindows()
        sys.exit(0)

def map_tokens_to_image(frame, tokens, bboxes, image=None, predictions=None):
    print(frame.shape)
    sf = 0
    print(f"{tokens[sf].shape=}")
    # current_token = np.sum(tokens[sf][0].numpy(), axis=0)

    current_token = torch.norm(tokens[sf][0], dim=0)
    print(current_token.shape)
    min_val = current_token.min()
    max_val = current_token.max()
    normalized_scores = (current_token - min_val) / (max_val - min_val + 1e-8)
    draw_and_display(frame, normalized_scores.numpy(), bboxes, image, predictions)
    if cv2.waitKey(0) == ord("q"):
        cv2.destroyAllWindows()
        sys.exit(0)
    # draw_plot(frame, masks)
def get_sparsity_mask(fpn_layer: torch.Tensor, threshold = 0.12):
    tokens = torch.norm(fpn_layer, dim=1)

    max_pool = nn.MaxPool2d(2,2)
    min_val = tokens.amin(dim=(-2, -1), keepdim=True)
    max_val = tokens.amax(dim=(-2, -1), keepdim=True)
    tokens = (tokens - min_val) / (max_val-min_val + 1e-8)
    # tokens = max_pool(tokens)
    # print(f"{tokens.shape=}")

    mask = tokens < threshold
    # print(f"{mask=}")

    data = torch.ones((mask.shape))
    # print(f"{data}")
    data.masked_fill_(mask, float(0))
    # print(f"{data}")
    return tokens

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
        mdl_str = "mdl."
        mdl_dict = {k.replace(mdl_str, ""): v 
                     for k, v in state_dict.items() if k.startswith(mdl_str)}
        module.mdl.load_state_dict(mdl_dict, strict=True)
    # if ckpt_path:
    #     print('Resuming only the weights instead of the full training state')
    #     ckpt = torch.load(ckpt_path, map_location='cpu')
    #     state_dict = ckpt["state_dict"]

    #     backbone_str = "mdl.backbone."
    #     backbone_dict = {k.replace(backbone_str, ""): v 
    #                  for k, v in state_dict.items() if k.startswith(backbone_str)}

    #     fpn_str = "mdl.fpn."
    #     fpn_dict = {k.replace(fpn_str, ""): v 
    #                  for k, v in state_dict.items() if k.startswith(fpn_str)}

    #     module.mdl.backbone.load_state_dict(backbone_dict, strict=True)
    #     module.mdl.fpn.load_state_dict(fpn_dict, strict=True)
    #     for param in module.mdl.backbone.parameters():
    #         param.requires_grad = False

    #     for param in module.mdl.fpn.parameters():
    #         param.requires_grad = False

    #         module.mdl.backbone.eval()
    #         module.mdl.fpn.eval()
    #     ckpt_path = None

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
    previous_states = None
    for batch in val_loader:
        data = batch['data']
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                ev_tensor_sequence = data[DataType.EV_REPR]
                sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
                is_first_sample = data[DataType.IS_FIRST_SAMPLE]
                image = None
                if DataType.IMAGE in data.keys():
                    data_image = data[DataType.IMAGE]
                token_mask_sequence = data.get(DataType.TOKEN_MASK, None)
                sequence_len = len(ev_tensor_sequence)
                batch_size = ev_tensor_sequence[0].shape[0]
                for tidx in range(sequence_len):
                    ev_tensors = ev_tensor_sequence[tidx]
                    ev_tensors = input_padder.pad_tensor_ev_repr(ev_tensors)
                    image = input_padder.pad_tensor_ev_repr(data_image[tidx])
                    bboxes = sparse_obj_labels[tidx][0]
                    new_bbs = None
                    if bboxes:
                        new_bb = bboxes.object_labels[:, 1:5]
                        new_bbs = np.vstack(new_bb)

                    preds, states, _ = module.mdl.backbone(ev_tensors, previous_states)
                    previous_states = states
                    # for i in range(len(preds)):
                    #     print(preds[i].shape)
                    preds = module.mdl.fpn(preds)

                    print(preds[0].shape)
                    print(preds[1].shape)
                    print(preds[2].shape)
                    preds[0][:, :, 33:48, :] = 0
                    preds[1][:, :, 0:24, :] = 0
                    preds[2][:, :, 0:12, :] = 0

                    output, _ = module.mdl.yolox_head(preds)
                    pred_processed = postprocess(prediction=output,
                                                 num_classes=mdl_config.head.num_classes,
                                                 conf_thre=mdl_config.postprocess.confidence_threshold,
                                                 nms_thre=mdl_config.postprocess.nms_threshold)
                    print(f"{pred_processed=}")
                    print(f"{new_bbs=}")
                    # for i in range(len(preds)):
                    #     preds[i] = max_pool(preds[i])
                    # preds = [preds[i] for i in [1, 2, 3, 4]]
                    map_tokens_to_image(ev_tensors.numpy(), preds, new_bbs, image, pred_processed)

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
