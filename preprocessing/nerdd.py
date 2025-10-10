import os
import re
import cv2
import random
from enum import Enum, auto
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from datetime import datetime, date
from utils.helper_functions import downsample_labels, get_stacked_histogram
from utils.nerd_event_loader import EventLoader
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

# Raw Data settings
SRC_PATH = Path("/archive/COMMON/NeRDD/DATA/")
OUT_PATH = Path("outpath")
SETS = ["train", "val"]

EVENT_DIR = "Event"
EVENT_FILE = "output_events.npz"
LABEL_FILE = "ev_rgb_coordinates.txt"
RGB_FOLDER = "RGB"

SAVE_EVENTS = True
SAVE_LABELS = True
SAVE_RGB = True
DELTA_TIME = 50000


# Ratios between Train and Validation
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2


class DataKeys(Enum):
    Labels = auto()
    InH5 = auto()
    RGB_Folder = auto()
    OutLabelDir = auto()
    OutEvReprDir = auto()
    OutRGBDir = auto()


def get_all_dirs(root: Path):
    all_folders = list()
    for archive in os.listdir(root):
        archive_path = root / archive
        if not archive_path.is_dir():
            continue
        for number in os.listdir(archive_path):
            # each element looks something like Archive_6/0
            number_path = archive_path / number
            if not number_path.is_dir():
                continue

            unique_dir_combination = (archive, number)
            all_folders.append(unique_dir_combination)

    return all_folders


def split_dataset(all_directories):
    random.seed(42)
    random.shuffle(all_directories)

    total_directories = len(all_directories)
    train_size = int(TRAIN_RATIO * total_directories)

    directories = {}
    directories["train"] = all_directories[:train_size]
    directories["val"] = all_directories[train_size:]

    return directories


def create_sequence(source_dir_parent, source_tuple, target_dir):
    assert (
        len(source_tuple) == 2
    ), f"Expect source_dir_name to be Tuple (Archive_#/#) but is {source_tuple}"
    save_dir_name = source_tuple[0] + "_" + source_tuple[1]
    input_dir_name = os.path.join(source_tuple[0], source_tuple[1])
    out_path = target_dir / save_dir_name

    out_labels_path = out_path / "labels"
    out_ev_repr_path = out_path / "events"
    out_rgb_path = out_path / "rgbs"

    os.makedirs(out_path, exist_ok=True)
    os.makedirs(out_ev_repr_path, exist_ok=True)
    os.makedirs(out_labels_path, exist_ok=True)
    os.makedirs(out_rgb_path, exist_ok=True)

    event_file = source_dir_parent / input_dir_name / EVENT_DIR / EVENT_FILE
    label_file = source_dir_parent / input_dir_name / LABEL_FILE
    rgb_folder = source_dir_parent / input_dir_name / RGB_FOLDER

    assert event_file.exists()
    assert label_file.exists()

    sequence_data = {
        DataKeys.Labels: label_file,
        DataKeys.InH5: event_file,
        DataKeys.RGB_Folder: rgb_folder,
        DataKeys.OutLabelDir: out_labels_path,
        DataKeys.OutEvReprDir: out_ev_repr_path,
        DataKeys.OutRGBDir: out_rgb_path,
    }

    return sequence_data

def get_matching_image_idx(image_ts: np.ndarray, frame_timestamp
                           , max_distance_to_timestamp):
    time_diff = np.abs(image_ts - frame_timestamp)
    min_index = np.argmin(time_diff)
    return min_index

def get_matching_labels(
    unique_labels: np.ndarray, frame_timestamp, max_distance_to_timestamp
):
    labels_timestamps = unique_labels["t"]
    time_diff = np.abs(labels_timestamps - frame_timestamp)
    valid_bbox_indices = np.where(time_diff <= max_distance_to_timestamp)[0]

    if len(valid_bbox_indices) == 0:
        label_type = [
            ("t", "i8"),
            ("x", "f4"),
            ("y", "f4"),
            ("w", "f4"),
            ("h", "f4"),
            ("class_id", "i1"),
            ("class_confidence", "f4"),
        ]
        labels_new = np.empty(0, dtype=label_type)
        return labels_new

    min_time_diff = np.min(time_diff[valid_bbox_indices])
    closest_indices = valid_bbox_indices[time_diff[valid_bbox_indices] == min_time_diff]
    closest_bboxes = unique_labels[closest_indices]
    return closest_bboxes

def timesamp_from_filename(today_str, file_name):
    match = re.search(r"\d{2}:\d{2}:\d{2}\.\d+", file_name)
    file_time = match.group()
    datetime_str = f"{today_str} {file_time}"
    dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
    return dt.timestamp()

def load_labels(path: Path):
    with open(path, "r") as f:
        data = f.read()
    data_clean = re.sub(r"[,:]", "", data)
    from io import StringIO

    labels = np.genfromtxt(StringIO(data_clean))
    # convert seconds to microseconds
    labels[:, 0] *= 1000 * 1000

    # convert min_x, min_y, max_x, max_y to min_x, min_y, widht, height
    labels[:, 3:] -= labels[:, 1:3]

    # bring labels into usually used structured array
    label_type = [
        ("t", "i8"),
        ("x", "f4"),
        ("y", "f4"),
        ("w", "f4"),
        ("h", "f4"),
        ("class_id", "i1"),
        ("class_confidence", "f4"),
    ]
    labels_new = np.empty(len(labels), dtype=label_type)
    labels_new["t"] = labels[:, 0]
    labels_new["x"] = np.clip(labels[:, 1], 0, WIDTH)
    labels_new["y"] = np.clip(labels[:, 2], 0, HEIGHT)
    labels_new["w"] = np.clip(labels[:, 3], 0, WIDTH - labels_new['x'])
    labels_new["h"] = np.clip(labels[:, 4], 0, HEIGHT - labels_new['y'])

    # there are only drones in the dataset so assign id 0
    labels_new["class_id"] = np.zeros_like(labels[:, 0])
    # we do not have confidence values so let's say we are 100% sure it's a drone
    labels_new["class_confidence"] = np.ones_like(labels[:, 0])
    return labels_new

def process_images(rgb_dir):
    today_str = date.today().strftime("%Y-%m-%d")
    rgb_files = os.listdir(rgb_dir)
    # fix sort to be consistent
    rgb_files.sort()
    start_time = timesamp_from_filename(today_str, rgb_files[0]) * 1000 * 1000
    file_times = {}
    file_time = 0
    for img_name in rgb_files:
        image_path = rgb_dir / img_name
        # file_time = timesamp_from_filename(today_str, img_name)
        # file_time *= 1000 * 1000
        # file_time -= start_time
        file_time += 33333
        # this works as there are no double images as this would not work
        # on a filesystem level
        file_times[file_time] = image_path

    return file_times

def process_sequence(sequence):
    events_file = sequence[DataKeys.InH5]
    label_file = sequence[DataKeys.Labels]
    out_labels_dir = sequence[DataKeys.OutLabelDir]
    out_events_dir = sequence[DataKeys.OutEvReprDir]
    rgb_folder = sequence[DataKeys.RGB_Folder]
    out_rgb_dir = sequence[DataKeys.OutRGBDir]

    event_loader = EventLoader(events_file)
    data_len = len(event_loader)
    if SAVE_EVENTS:
        for i in range(data_len):
            event_name = f"event_{i}"
            out_event_path = out_events_dir / Path(f"{event_name}.npz")
            events = event_loader.get_item(i)
            stacked_histogram = get_stacked_histogram(
                STACKEDHISTOGRAM, events, DOWNSAMPLE
            ).numpy()
            np.savez_compressed(out_event_path, stacked_histogram)
    if SAVE_LABELS:
        labels = load_labels(label_file)
        # we filter right now by unique labels as there are some bounding boxes with the
        # exact same timestamp and location
        # this assumes that there is only one drone in every frame for all frames
        # this can be a problem if there a bounding boxes for multiple drones
        # we could filter by how close the bounding boxes are than
        _, unique_indecies = np.unique(labels["t"], return_index=True)
        labels = labels[unique_indecies]
        for i in range(data_len):
            label_name = f"label_{i}"
            out_path_label = out_labels_dir / Path(f"{label_name}.npz")
            start = event_loader.get_start_time(i)
            end = event_loader.get_end_time(i)
            frame_timestamp = (start + end) / 2
            matching_labels = get_matching_labels(labels, frame_timestamp, DELTA_T // 2)
            downsampled_labels = downsample_labels(matching_labels, DOWNSAMPLE)
            np.savez_compressed(out_path_label, downsampled_labels)
    if SAVE_RGB:
        image_ts_path = process_images(rgb_folder)
        image_timestamps = np.asarray(list(image_ts_path.keys()))
        image_time = 0
        for i in range(data_len):
            rgb_name = f"rgb_{i}"
            out_rgb_path = out_rgb_dir / Path(f"{rgb_name}.npz")
            start = event_loader.get_start_time(i)
            end = event_loader.get_end_time(i)
            frame_timestamp = (start + end) / 2
            img_key = get_matching_image_idx(image_timestamps, frame_timestamp, DELTA_T // 2)
            image_path = image_ts_path[image_timestamps[img_key]]
            rgb = cv2.imread(str(image_path), cv2.COLOR_BGR2RGB)
            if DOWNSAMPLE > 1:
                rgb = cv2.resize(rgb, 
                                   (WIDTH//DOWNSAMPLE, HEIGHT//DOWNSAMPLE),
                                   interpolation=cv2.INTER_AREA)
            np.savez_compressed(out_rgb_path, rgb)


    

def setup_data() -> List[DataKeys]:
    all_directories = get_all_dirs(SRC_PATH)
    directories = split_dataset(all_directories)

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
