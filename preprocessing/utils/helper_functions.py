from utils.representations import StackedHistogram
import torch
import numpy as np


def downsample_labels(labels, downsample_factor):
    output_labels = labels.copy()
    if (len(output_labels) > 0):
        output_labels['x'] /= downsample_factor
        output_labels['y'] /= downsample_factor
        output_labels['w'] /= downsample_factor
        output_labels['h'] /= downsample_factor
    
    return output_labels

def get_stacked_histogram(stacked_histogram: StackedHistogram, events: np.ndarray, downsample: float):
    events['x'] //= downsample
    events['y'] //= downsample
    # ToDo: is int32 the way to go check this
    x = torch.from_numpy(events['x'].astype(np.int32))
    y = torch.from_numpy(events['y'].astype(np.int32))
    p = torch.from_numpy(events['p'].astype(np.int32))
    t = torch.from_numpy(events['t'].astype(np.int32))

    tensor = stacked_histogram.construct(x, y, p, t)
    return tensor
