# Visualize preprocessed data just drop a path to the preprocessed data

import numpy as np
import torch
import os
import cv2
import sys
import random
import threading

from queue import Queue
from enum import Enum, auto

from pathlib import Path
from typing import List

sys.path.append("./utils")
from draw_stuff import draw_and_display, extract_bounding_boxes


class State(Enum):
    SKIP = auto()
    NEXT = auto()
    BACKWARDS = auto()
    SKIP_FOLDER = auto()
    NOP = auto()

def draw_and_display_threaded(queue):
    while True:
        try:
            events, labels = queue.get()
            if events is None and labels is None:
                cv2.destroyAllWindows()
                return
            draw_and_display(events, labels)
            cv2.waitKey(200)
        except Exception as e:
            print(f"Error in displaying: {e}")
            cv2.waitKey(200)

def parse_user_input(input:str) -> State:
    state = None
    skip_items = 0
    if len(input) == 0:
        state = State.NOP
    elif input[0] == 's':
        state = State.SKIP
        skip_items = int(input[1:])
    elif input[0] == 'n':
        state = State.NEXT
        skip_items = int(input[1:])
    elif input[0] == 'f':
        state = State.SKIP_FOLDER
    elif input[0] == 'b':
        state = State.BACKWARDS


    return (state, skip_items)


def visualize(directories: List[Path], queue, is_random: bool = False):
    if is_random:
        print("Random is set shuffeling...")
        random.shuffle(directories)

    dir_len = len(directories)
    file_len = 0
    dir_count = 0
    file_count = 0
    items_to_skip = 0
    backwards = 1
    while True:
        dir_count = dir_count % dir_len
        dir = directories[dir_count]
        print(dir)
        events_path = dir / "events"
        labels_path = dir / "labels"


        event_files = os.listdir(events_path)
        label_files = os.listdir(labels_path)
        assert len(event_files) == len(label_files)

        file_len = len(event_files)
        print(f"Directory: {dir}, Element_count: {file_len}")
        if backwards == 1:
            file_count = 0
        else:
            file_count = file_len - 1

        while file_count >= 0 and file_count < file_len:
            if items_to_skip <= 0:
                items_to_skip = 0
                skip_files = False
            if not skip_files:
                print(f"Event{event_files[file_count]}, Label{label_files[file_count]}")
                event_path = events_path / event_files[file_count]
                label_path = labels_path / label_files[file_count]

                events = np.load(event_path)['arr_0']
                labels = np.load(label_path)['arr_0']

                labels = extract_bounding_boxes(labels).astype(np.int32)

                queue.put((events, labels))
            file_count += 1 * backwards

            if items_to_skip > 0:
                items_to_skip -= 1
            else:
                user_input = input()
                if user_input == "q":
                    queue.put((None, None))
                    return
                else:
                    state, items_to_skip = parse_user_input(user_input)
                    if state is State.SKIP_FOLDER:
                            break
                    elif state is State.SKIP:
                        skip_files = True
                    elif state is State.NEXT:
                        skip_files = False
                    elif state is State.BACKWARDS:
                        backwards = backwards * -1
                    else:
                        items_to_skip = 0
                        skip_files = False
        dir_count += 1 * backwards


def get_directories(path: Path) -> List[Path]:
    directories = []
    for dir in os.listdir(path):
        dir_path = path / dir
        directories.append(dir_path)
    return directories


if __name__ == "__main__":
    print("Insert command into command line to see next Image")
    print("Press enter to see the next frame")
    print("b reverses the direction for all commands")
    print("n10 displays the next 10 images where 10 can be replaced by any number")
    print("s10 same as n10 but instead of displaying skips ahead(faster)")
    print("f skips to the end of the current folder and displays the next folder")
    if len(sys.argv) < 2:
        print(f"pls provide path to dataset folder")

    path = Path(sys.argv[1])
    assert path.exists()
    is_random = False
    if len(sys.argv) > 2:
        is_random = True

    train_path = path / "train"
    val_path = path / "val"
    events_path = path / "events"
    directories = []
    if train_path.exists():
        train_dirs = get_directories(train_path)
        val_dirs = get_directories(val_path)
        directories.extend(train_dirs)
        directories.extend(val_dirs)
    elif events_path.exists():
        # paths = get_directories(path)
        directories.append(path)
    else:
        paths = get_directories(path)
        directories.extend(paths)

    
    queue = Queue(maxsize=0)
    opencv_thread = threading.Thread(target=draw_and_display_threaded,
                                     args=(queue,),)
    opencv_thread.start()
    visualize(directories, queue, is_random)
