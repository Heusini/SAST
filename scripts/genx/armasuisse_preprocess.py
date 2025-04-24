from util.representations import StackedHistogram
from util.event_loader import EventLoader
import numpy as np
import random
import torch
import json
import os

USE_GPU = True

HEIGHT = 720
WIDTH = 1280
DOWNSAMPLE = 1

BINS = 3
CHANNELS_PER_BIN = 2
DELTA_T = 50000

SEQUENCE_LENGTH = 5

src_path = '/st_stephan_raw'
target_path = '/sh_st_stephan'
sets = ['train', 'val']

event_file = 'events_left_final.h5'
label_file = 'labels_events_left.npy'
matching_file = 'frames_ts.csv'

TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

USE_GPU = True

class_mapping = {1: 0.0}

categories = [
    {
        "name": "drone",
        "id": 0
    }
]

def split_dataset(path):
    all_directories = os.listdir(path)
    random.seed(42)
    random.shuffle(all_directories)

    total_directories = len(all_directories)
    train_size = int(TRAIN_RATIO * total_directories)

    directories = {}
    directories['train'] = all_directories[:train_size]
    directories['val'] = all_directories[train_size:]

    return directories

def preprocess():
    directories = split_dataset(src_path)

    sh = StackedHistogram(bins=BINS, height=HEIGHT//DOWNSAMPLE, width=WIDTH//DOWNSAMPLE, count_cutoff=10)

    for set in sets:
        print("Processing", set, "set")
        act_folder_nr = 1
        folder_count = str(len(directories[set]))

        file_path = "{}/data/{}".format(target_path, set)
        json_file = "{}/{}.json".format(target_path, set)

        res_file = {
            "categories": categories,
            "images": [],
            "annotations": []
        }

        image_id = 0
        annot_id = 0

        # Std + Mean variables
        num_pixels = 0
        channel_sums = torch.zeros(BINS * CHANNELS_PER_BIN)
        channel_squared_sums = torch.zeros(BINS * CHANNELS_PER_BIN)
        pixels_per_channel = (WIDTH//DOWNSAMPLE)*(HEIGHT//DOWNSAMPLE)

        # Delete json file if exists
        if os.path.exists(json_file):
            os.remove(json_file)

        # Delete data if exists
        print('Deleting existing files...')
        for filename in os.listdir(file_path):
            rem_file = file_path + '/' + filename
            os.remove(rem_file)

        for folder in directories[set]:
            print("Processing", folder, "("+str(act_folder_nr),"of",folder_count+")")
            act_folder_nr += 1

            # Get names of h5 and npy files
            h5_file = src_path + '/' + folder + '/' + event_file
            npy_file = src_path + '/' + folder + '/' + label_file
            csv_file = src_path + '/' + folder + '/' + matching_file

            # Get object bounding boxes
            bboxes = np.load(npy_file)
            matching = np.genfromtxt(csv_file, delimiter=',', skip_header=True)

            # Create event loader
            try:
                event_loader = EventLoader(h5_file, matching, use_gpu=USE_GPU, delta_t=DELTA_T)
            except Exception as e:
                print(f"Error opening {h5_file}: {e}")
                continue

            i = 0
            files_sequence = []
            while True:
                events = event_loader.get_item(i)
                file_name = folder + '_' + str(i)
                data_path = file_path + "/" + file_name

                if not isinstance(events, np.ndarray):
                    break

                if events.size > 0:
                    events['x'] //= DOWNSAMPLE
                    events['y'] //= DOWNSAMPLE
                    x = torch.from_numpy(events['x'].astype(np.int32))
                    y = torch.from_numpy(events['y'].astype(np.int32))
                    p = torch.from_numpy(events['p'].astype(np.int32))
                    t = torch.from_numpy(events['t'].astype(np.int32))

                    tensor = sh.construct(x, y, p, t)

                    if set == 'train':
                        num_pixels += pixels_per_channel
                        channel_sums += tensor.sum(dim=(1, 2))
                        channel_squared_sums += (tensor ** 2).sum(dim=(1, 2))

                    torch.save(tensor, data_path)
                    files_sequence.append(file_name)

                if len(files_sequence) == SEQUENCE_LENGTH:
                    filtered_boxes = bboxes[bboxes['frame'] == i]
                    if len(filtered_boxes) > 0:
                        img_elem = {
                                "file_name": file_name,
                                "sequence_id": act_folder_nr,
                                "sequence": json.dumps(files_sequence),
                                "width": WIDTH // DOWNSAMPLE,
                                "height": HEIGHT // DOWNSAMPLE,
                                "id": image_id,
                            }

                        res_file["images"].append(img_elem)

                        for bbox in filtered_boxes:
                            annot_elem = {
                                "id": annot_id,
                                "bbox": [
                                    float(bbox['x'] // DOWNSAMPLE),
                                    float(bbox['y'] // DOWNSAMPLE),
                                    float(bbox['w'] // DOWNSAMPLE),
                                    float(bbox['h'] // DOWNSAMPLE)
                                ],
                                "image_id": image_id,
                                "ignore": 0,
                                "category_id": class_mapping[bbox['class_id']],
                                "iscrowd": 0,
                                "area": float(bbox['w'] * bbox['h'])
                            }

                            res_file["annotations"].append(annot_elem)
                            annot_id += 1

                        files_sequence.pop(0)
                        image_id += 1
                    else:
                        # reset files_sequence
                        files_sequence = []

                i += 1

        # Calc std + mean
        if set == 'train':
            channel_means = channel_sums / num_pixels
            channel_variances = (channel_squared_sums / num_pixels) - (channel_means ** 2)
            channel_stds = torch.sqrt(channel_variances)
            mean = channel_means.tolist()
            std = channel_stds.tolist()
            print(f"Mean of train set: {mean}")
            print(f"Std of train set: {std}")

        with open(json_file, "w") as f:
            json_str = json.dumps(res_file)
            f.write(json_str)


if __name__ == "__main__":
    preprocess()
