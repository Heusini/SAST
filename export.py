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
    dummy_img = torch.randint(0, 10, (20, 384, 640), dtype=torch.float32)
    inps = dummy_img.to(device)
    inps = torch.stack([inps for _ in range(1)])
    return inps

def make_infer_image(device="cuda"):
    dummy_img = torch.randint(0, 10, (3, 384, 640), dtype=torch.float32)
    inps = dummy_img.to(device)
    inps = torch.stack([inps for _ in range(1)])
    return inps

def export_onnx(model, input_names, input_tensors, output_names, dynamic_axes):
    output_file = '/home/sheusinger/sast.onnx'

    torch.onnx.export(
        model,
        input_tensors,
        output_file,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=True,
        verbose=True,
        opset_version=17,
        dynamic_axes=dynamic_axes
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
    config = OmegaConf.load('config/detect_lwdetr.yaml')

    # device for export onnx
    device = torch.device("cpu")
    
    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

     # Load the model from the checkpoint
    module = fetch_model_module(config=config)
    # model = module.load_from_checkpoint(config.checkpoint, **{'full_config': config})
    model = module

    # Make sure the model is in evaluation mode
    model.eval()

    input_tensors1 = make_infer_event(device)
    input_tensors2 = make_infer_image(device)
    input_tensors = (input_tensors1, input_tensors2)
    input_names = ['input']
    output_names = ['dets']
    dynamic_axes = None

    output_file = export_onnx(model, input_names, input_tensors, output_names, dynamic_axes)
    output_file = onnx_simplify(output_file, input_names, input_tensors)
    output_file = trtexec(output_file)

if __name__ == '__main__':
    main()
