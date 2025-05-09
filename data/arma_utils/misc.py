# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import datetime
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict, deque
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist

# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
from PIL import Image, ImageDraw
from torch import Tensor
from tqdm import tqdm

import wandb
from data.arma_utils import box_ops

if float(torchvision.__version__.split(".")[1]) < 7.0:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,), dtype=torch.uint8, device="cuda"
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

        self.print_names = [
            "epoch",
            "lr",
            "loss",
            "class_error",
            "loss_bbox",
            "loss_giou",
            "loss_ce",
        ]

        self.pbar = None

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, epoch=-1, log=True):
        if self.pbar == None:
            self.pbar = tqdm(total=len(iterable))

        i = 0

        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")

        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                metrics = {}
                for name, meter in self.meters.items():
                    if name in self.print_names:
                        metrics[name] = meter.value
                if log:
                    # Calculate global step
                    global_step = int(i + epoch * len(iterable))
                    wandb.log({"train/step": metrics}, step=global_step)

                if epoch == -1:
                    self.pbar.set_description("Evaluation")
                else:
                    self.pbar.set_description("Epoch {}".format(epoch))

                self.pbar.update(print_freq)
                self.pbar.set_postfix(loss=self.meters["loss"])

            i += 1
            end = time.time()


class TokenSelector(object):
    def __init__(self, patch_size, sparse_threshold, sparse_min_tokens):
        self.patch_size = patch_size
        self.sparse_threshold = sparse_threshold
        self.sparse_min_tokens = sparse_min_tokens

    def get_active_tokens(self, x):
        unfolded_tensor = (
            x.unsqueeze(0)
            .unfold(2, self.patch_size, self.patch_size)
            .unfold(3, self.patch_size, self.patch_size)
        )
        event_count = (unfolded_tensor > 0).sum(dim=(-1, -2))
        event_count_per_patch = event_count.sum(dim=1)
        # Create mask with threshold
        mask = event_count_per_patch > self.sparse_threshold
        mask = mask.reshape(1, -1)[0]

        true_count = mask.sum().item()

        if true_count < self.sparse_min_tokens:
            sorted_indices = event_count_per_patch[0].flatten().argsort(descending=True)
            mask_indices = sorted_indices[: self.sparse_min_tokens]
            mask.flatten()[mask_indices] = True

        active_tokens = []
        dims = (unfolded_tensor.shape[2], unfolded_tensor.shape[3])
        mask.reshape(dims[0], dims[1])

        for i in range(dims[0]):
            for j in range(dims[1]):
                if mask[i * dims[0] + j]:
                    active_tokens.append(
                        [
                            j * self.patch_size,
                            i * self.patch_size,
                            j * self.patch_size + self.patch_size,
                            i * self.patch_size + self.patch_size,
                        ]
                    )

        return active_tokens


def save_image(
    samples, outputs, posprocess, log_str, targets=None, token_selector=None
):
    # Create image
    summed_hist = torch.sum(samples.tensors[-1], axis=0)
    image = (
        255
        * (summed_hist - torch.min(summed_hist))
        / (torch.max(summed_hist) - torch.min(summed_hist))
    )
    image = Image.fromarray(image.cpu().numpy().astype(np.uint8))

    # Resize to non-square
    image_rgb = Image.merge(
        "RGBA", (image, image, image, Image.new("L", image.size, 255))
    )
    image_rgb = image_rgb.resize((640, 360))
    image_draw = ImageDraw.Draw(image_rgb, "RGBA")
    results = posprocess(outputs)

    # Draw predictions
    keep = results[-1]["scores"] > 0.5
    for bbox in results[-1]["boxes"][keep]:
        x_min, y_min, x_max, y_max = map(int, bbox)
        image_draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=(0, 255, 0, 255))

    # Draw ground truth
    if targets:
        for target in targets[-1]["boxes"]:
            target = box_ops.box_cxcywh_to_xyxy(target)
            scale_fct = torch.tensor([640, 360, 640, 360]).to(target.device)
            target *= scale_fct

            x_min, y_min, x_max, y_max = target

            image_draw.rectangle(
                [(x_min, y_min), (x_max, y_max)], outline=(255, 0, 0, 255)
            )

    # Draw selected tokens
    if token_selector is not None:
        active_tokens = token_selector.get_active_tokens(samples.tensors[-1])
        rectangle_image = Image.new(
            "RGBA", image_rgb.size, (0, 0, 0, 0)
        )  # Transparent black
        image_draw = ImageDraw.Draw(rectangle_image)
        for active_token in active_tokens:
            active_token[1] = active_token[1] / 640 * 360
            active_token[3] = active_token[3] / 640 * 360

            image_draw.rectangle(
                [
                    (active_token[0], active_token[1]),
                    (active_token[2], active_token[3]),
                ],
                fill=(255, 255, 255, int(255 * 0.4)),
            )

        image_rgb = Image.alpha_composite(image_rgb, rectangle_image)

    wandb.log({log_str: wandb.Image(np.array(image_rgb))})


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message

def collate_fn_split(batch):
    batch = list(zip(*batch))
    _, channels, height, width = batch[0][0].shape
    _, mask_size = batch[2][0].shape

    data = torch.stack(batch[0]).reshape(-1, channels, height, width)
    # print(len(batch[1]))
    # print(batch[1][0][1])
    target = np.stack(batch[1]).reshape(-1)
    sparsity = torch.stack(batch[2]).reshape(-1, mask_size)

    mask = torch.zeros((data.shape[0], height, width), dtype=bool)

    batch[0] = NestedTensor(data, mask)
    batch[1] = tuple(target)
    batch[2] = tuple(sparsity)
    return tuple(batch)



# batch is tensors + sparsity mask (i think)?
def collate_fn(batch):
    # batch is a tuple of three values of tensor (image), boxes and something else
    # we zip these tuple into a list where the first list element is a list of lists
    # tensors of all the
    batch = list(zip(*batch))
    # print()
    # print()
    # print(len(batch[0]))
    # sys.exit(0)
    # sys.exit(0)

    # len(tensor) = batch_size
    # those tensors are then converted with nested_tensor_from_tensor_list
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    # print(type(batch[1][7]))
    # <class 'dict'>
    # print(batch[1][7])
    # ^
    # {'boxes': tensor([[0.5674, 0.5114, 0.0177, 0.0189]]), 'labels': tensor([0]), 'image_id': tensor([1091]), 'orig_size': tensor([360, 640]), 'size': tensor([640, 640])}
    # torch.Size([8, 6, 640, 640])


    # returns padded_tensors pad_mask + sparsity_mask
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def nested_tensor_fuck_it(tensor_list):
    pass


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    # checks if the first tensor in the tensor list has dim=3
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images

        # we get the max size for each axis
        # example max_size = [6, 640, 640]
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))

        # this concatinates two lists so we have [len(tensor_list), max_size] where max_size
        # = max_size = [6, 640, 640]
        # result is something like this [8, 6, 640, 640]
        batch_shape = [len(tensor_list)] + max_size
        # batch, channel, height, width (name your fucking variables better)
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)

        # mask is uesd to indcate which padded areas should not be used in the
        # transformer?
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)

        # zip goes over first dimension in this case the batch sizes and than iterates over
        # each element
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")

    # output is that all tensors have the same size and the tensors have zero padding
    # the mask is zero where the image is and ones where their is padding
    # tensor has the shape [batch, channel, height, weight]
    # mask has the shape [batch, height, weight]
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(
            img, (0, padding[2], 0, padding[1], 0, padding[0])
        )
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(
            m, (0, padding[2], 0, padding[1]), "constant", 1
        )
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__.split(".")[1]) < 7.0:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(
            input, size, scale_factor, mode, align_corners
        )


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
