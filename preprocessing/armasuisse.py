import os
import sys
import random
from enum import Enum, auto
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import List

import numpy as np
import torch
import cv2
from tqdm import tqdm

from utils.arma_event_loader import EventLoader
from utils.helper_functions import downsample_labels, get_stacked_histogram
from utils.representations import StackedHistogram

# Processing Settings
NUM_CPUS = 8

# Input
HEIGHT = 720
WIDTH = 1280
DOWNSAMPLE = 2

# StackedHistogram Settings
BINS = 10
CHANNELS_PER_BIN = 2
DELTA_T = 50000
COUNT_CUTOFF = 10
STACKEDHISTOGRAM = StackedHistogram(
    bins=BINS,
    height=HEIGHT // DOWNSAMPLE,
    width=WIDTH // DOWNSAMPLE,
    count_cutoff=COUNT_CUTOFF,
)


# No Option right now to create sequences of data do this in the DataLoader
# SEQUENCE_LENGTH = 5

# Raw Data settings
SRC_PATH = Path("/datasets/armasuisse/StStephan/drone/")
OUT_PATH = Path("outpath")
SETS = ["train", "val"]

SAVE_RGB = True
SAVE_EVENTS = True
SAVE_LABELS = True


EVENT_FILE = "events_left_final.h5"
LABEL_FILE = "labels_events_left.npy"
MATCHING_FILE = "frames_ts.csv"
RGB_FOLDER = "frames"
HOMOGRAPHY_FILE = "Projection_rgb_to_events_left.npy"

# Ratios between Train and Validation
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2


class DataKeys(Enum):
    Labels = auto()
    InH5 = auto()
    Matching = auto()
    RGB_Folder = auto()
    Homography = auto()
    OutLabelDir = auto()
    OutEvReprDir = auto()
    OutRGBDir = auto()


def split_dataset(path):
    all_directories = os.listdir(path)
    random.seed(42)
    random.shuffle(all_directories)

    total_directories = len(all_directories)
    train_size = int(TRAIN_RATIO * total_directories)

    directories = {}
    directories["train"] = all_directories[:train_size]
    directories["val"] = all_directories[train_size:]

    return directories


def create_sequence(source_dir_parent, source_dir_name, target_dir):
    out_path = target_dir / source_dir_name

    out_labels_path = out_path / "labels"
    out_ev_repr_path = out_path / "events"
    out_rgb_path = out_path / "rgbs"

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_ev_repr_path, exist_ok=True)
    os.makedirs(out_labels_path, exist_ok=True)
    os.makedirs(out_rgb_path, exist_ok=True)

    h5f_path = source_dir_parent / source_dir_name / EVENT_FILE
    label_file = source_dir_parent / source_dir_name / LABEL_FILE
    matching_file = source_dir_parent / source_dir_name / MATCHING_FILE
    rgb_folder = source_dir_parent / source_dir_name / RGB_FOLDER
    homography_file = source_dir_parent / source_dir_name / HOMOGRAPHY_FILE

    sequence_data = {
        DataKeys.Labels: label_file,
        DataKeys.InH5: h5f_path,
        DataKeys.Matching: matching_file,
        DataKeys.RGB_Folder: rgb_folder,
        DataKeys.Homography: homography_file,
        DataKeys.OutLabelDir: out_labels_path,
        DataKeys.OutEvReprDir: out_ev_repr_path,
        DataKeys.OutRGBDir: out_rgb_path,
    }

    return sequence_data


def format_boxes(boxes: np.ndarray):
    new_boxes = boxes[["frame", "x", "y", "w", "h", "class_id", "class_confidence"]]
    dtype = [
        ("t", "i8"),
        ("x", "f4"),
        ("y", "f4"),
        ("w", "f4"),
        ("h", "f4"),
        ("class_id", "u1"),
        ("class_confidence", "f4"),
    ]

    boxes = new_boxes.astype(dtype, copy=True)
    boxes["class_id"] -= 1
    return boxes


def process_sequence(sequence):
    events_file = sequence[DataKeys.InH5]
    label_file = sequence[DataKeys.Labels]
    matching_file = sequence[DataKeys.Matching]
    rgb_folder = sequence[DataKeys.RGB_Folder]
    out_labels_dir = sequence[DataKeys.OutLabelDir]
    out_events_dir = sequence[DataKeys.OutEvReprDir]
    out_rgb_dir = sequence[DataKeys.OutRGBDir]

    homography_file = sequence[DataKeys.Homography]


    matching = np.genfromtxt(matching_file, delimiter=",", skip_header=True)

    if SAVE_EVENTS:
        event_loader = EventLoader(events_file, matching, DELTA_T)
        for i in range(len(matching)):
            event_name = f"event_{i}"
            out_path_event = out_events_dir / Path(f"{event_name}.npz")
            events = event_loader.get_item(i)
            stacked_histogram = get_stacked_histogram(
                STACKEDHISTOGRAM, events, DOWNSAMPLE
            ).numpy()
            np.savez_compressed(out_path_event, stacked_histogram)

    if SAVE_LABELS:
        bboxes = np.load(label_file)
        for i in range(len(matching)):
            label_name = f"label_{i}"
            out_label_path = out_labels_dir / Path(f"{label_name}.npz")
            frame_boxes = bboxes[bboxes["frame"] == i]
            formated_boxes = format_boxes(frame_boxes)
            downsampled_boxes = downsample_labels(formated_boxes, DOWNSAMPLE)
            np.savez_compressed(out_label_path, downsampled_boxes)
    if SAVE_RGB:
        rgb_names = os.listdir(rgb_folder)
        for rgb_name in rgb_names:
            rgb_path = rgb_folder / rgb_name
            num = int(rgb_name.split('.')[0])
            rgb_name = f"rgb_{num}"
            out_rgb_path = out_rgb_dir / Path(f"{rgb_name}.npz")
            rgb = cv2.imread(str(rgb_path), cv2.COLOR_BGR2RGB)

            # move frame into event camera
            homography = np.load(homography_file)
            rgb = cv2.warpPerspective(rgb, homography, (WIDTH, HEIGHT))

            if DOWNSAMPLE > 1:
                rgb = cv2.resize(rgb, 
                                   (WIDTH//DOWNSAMPLE, HEIGHT//DOWNSAMPLE),
                                   interpolation=cv2.INTER_AREA)
            np.savez_compressed(out_rgb_path, rgb)

def setup_data() -> List[DataKeys]:
    directories = split_dataset(SRC_PATH)

    seq_data_list = list()
    for set in SETS:
        print(f"Setting up dataset {set}...")
        for folder in directories[set]:
            out_path = OUT_PATH / set
            sequence_data = create_sequence(SRC_PATH, folder, out_path)
            seq_data_list.append(sequence_data)

    return seq_data_list


def preprocess(seq_data_list):
    print(f"Start preprocessing...")
    if NUM_CPUS > 1:
        chunksize = 1
        func = partial(process_sequence)

        with get_context("spawn").Pool(NUM_CPUS) as pool:
            with tqdm(total=len(seq_data_list), desc="sequences") as pbar:
                for _ in pool.imap_unordered(
                    func, iterable=seq_data_list, chunksize=chunksize
                ):
                    pbar.update()

    else:
        for sequence in tqdm(seq_data_list, desc="sequences"):
            process_sequence(sequence)


if __name__ == "__main__":
    seq_data_list = setup_data()
    preprocess(seq_data_list)
