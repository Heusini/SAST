# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
export ONNX model and TensorRT engine for deployment
"""
import os
import ast
import random
import argparse
import subprocess
from pathlib import Path

import onnx
import torch
import onnxsim
import numpy as np
from PIL import Image

from modules.utils.fetch import fetch_model_module
from utils.timers import CudaTimer as CudaTimer
from omegaconf import OmegaConf
import torch

from utils.optimizer import OnnxOptimizer
from config.modifier import dynamically_modify_train_config


def run_command_shell(command, dry_run:bool = False) -> int:
    if dry_run:
        print("")
        print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} {command}")
        print("")
        status = 0
    else:
        status = subprocess.call(command, shell=True)
    return status


def make_infer_event(device="cuda"):
    dummy_img = torch.randint(0, 1, (1, 20, 384, 640), dtype=torch.uint8)
    return dummy_img

def make_infer_image(device="cuda"):
    dummy_img = torch.rand((1, 3, 384, 640), dtype=torch.float32)
    return dummy_img

def export_onnx(model, input_names, input_tensors, output_names, dynamic_axes):
    output_file = './sast.onnx'

    model.eval()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            input_tensors,
            output_file,
            input_names=input_names,
            output_names=output_names,
            export_params=True,
            keep_initializers_as_inputs=False,
            training=torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            verbose=False,
            opset_version=17,
            # dynamo=False,
            # dynamic_axes=dynamic_axes
        )

    print(f'Successfully exported ONNX model: {output_file}')
    return output_file


def onnx_simplify(onnx_dir:str, input_names, input_tensors):
    sim_onnx_dir = onnx_dir.replace(".onnx", ".sim.onnx")
    
    if isinstance(input_tensors, torch.Tensor):
        input_tensors = [input_tensors]
    
    print(f'start simplify ONNX model: {onnx_dir}')
    opt = OnnxOptimizer(onnx_dir)
    opt.info('Model: original')
    opt.common_opt()
    opt.info('Model: optimized')
    opt.save_onnx(sim_onnx_dir)
    return sim_onnx_dir


def trtexec(onnx_dir:str) -> None:
    engine_dir = onnx_dir.replace(".onnx", f".engine")
    addition = "--useCudaGraph --useSpinWait --warmUp=500 --avgRuns=1000"
    verbose = ""
    command = " ".join([
        "trtexec",
            f"--onnx={onnx_dir}",
            f"--saveEngine={engine_dir}",
            f"--fp16", # --workspace=4096 
            f"{addition}",
            f"{verbose}"])

    status = run_command_shell(command)
    assert status == 0, f"error({status}) in infer command: {command}"
    print(f'Successfully serialized TensorRT engine: {engine_dir}')
    return engine_dir


def main():
     # Load the configuration file
    config = OmegaConf.load('config/detect.yaml')
    dynamically_modify_train_config(config)

    # device for export onnx
    device = torch.device("cpu")
    
    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



     # Load the model from the checkpoint
    module = fetch_model_module(config=config)
    ckpt_path = config.checkpoint
    # module = module.load_from_checkpoint(config.checkpoint, **{'full_config': config}, strict=False)

    print('Resuming only the weights instead of the full training state')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt["state_dict"]

    backbone_str = "mdl.backbone."
    backbone_dict = {k.replace(backbone_str, ""): v 
                 for k, v in state_dict.items() if k.startswith(backbone_str)}

    fpn_str = "mdl.fpn."
    fpn_dict = {k.replace(fpn_str, ""): v 
                 for k, v in state_dict.items() if k.startswith(fpn_str)}

    rgb_fpn_str = "mdl.rgb_fpn."
    rgb_dict = {k.replace(rgb_fpn_str, ""): v 
                 for k, v in state_dict.items() if k.startswith(rgb_fpn_str)}

    yolox_head_str = "mdl.yolox_head."
    yolox_dict = {k.replace(yolox_head_str, ""): v 
                 for k, v in state_dict.items() if k.startswith(yolox_head_str)}

    module.mdl.backbone.load_state_dict(backbone_dict, strict=True)
    module.mdl.fpn.load_state_dict(fpn_dict, strict=True)
    module.mdl.rgb_fpn.load_state_dict(rgb_dict, strict=True)
    module.mdl.yolox_head.load_state_dict(yolox_dict, strict=True)
    for param in module.mdl.backbone.parameters():
        param.requires_grad = False

    for param in module.mdl.fpn.parameters():
        param.requires_grad = False

    for param in module.mdl.rgb_fpn.parameters():
        param.requires_grad = False

    for param in module.mdl.yolox_head.parameters():
        param.requires_grad = False

    module.mdl.backbone.eval()
    module.mdl.fpn.eval()
    module.mdl.rgb_fpn.eval()
    module.mdl.yolox_head.eval()
    ckpt_path = None

    model = module

    # Make sure the model is in evaluation mode
    model.eval()

    input_tensors1 = make_infer_event(device)
    input_tensors2 = make_infer_image(device)

    out, loss, states = model(input_tensors1, input_tensors2)

    input_names = ['input', 'rgb_image']
    output_names = ['detections']
    for i, (h, c) in enumerate(states):
        input_names += [f'h_{i}', f'c_{i}']
        output_names += [f'h_{i}_out', f'c_{i}_out']

    input_tensors = (input_tensors1, input_tensors2, states)
    dynamic_axes = None

    output_file = export_onnx(model, input_names, input_tensors, output_names, dynamic_axes)
    output_file = onnx_simplify(output_file, input_names, input_tensors)
#    output_file = trtexec(output_file)

if __name__ == '__main__':
    main()
