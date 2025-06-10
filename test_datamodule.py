from modules.utils.fetch import fetch_data_module
import torch
import cv2
import hydra
import sys
import os
from tqdm import tqdm
import numpy as np
from omegaconf import DictConfig, OmegaConf
import bbox_visualizer as bbv

from data.utils.types import DataType, DatasetMode
from data.utils.object_labels import ObjectLabels
from data.utils.sparsely_batched_object_labels import SparselyBatchedObjectLabels
from data.arma_utils.armasuisse_augmented import ArmasuisseAugmented
from data.genx_utils.labels import ObjectLabelFactory as OLF

def draw_and_display(img_data, new_bb, window_name='window'):
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
    img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    if new_bb is not None:
        # new_bb = new_bb[:,1:]
        new_bb[:, 2:] += new_bb[:, :2]
        new_bb = new_bb.astype(np.int32)
        img = bbv.draw_multiple_rectangles(img, new_bb.tolist(), thickness=1)

    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
    cv2.imshow(window_name, img)

def draw_and_wait(img_data, new_bb):
    draw_and_display(img_data, new_bb)
    if cv2.waitKey(0) == ord("q"):
        cv2.destroyAllWindows()
        sys.exit(0)


def test_objectlabelfactory():
    path = "/archive/sheusinger/new_size_st_stephan/train/2024_01_10_112814_drone_002/labels/labels_0.npy"
    label = np.load(path)
    print(label)
    print(label.shape)
    factory = ObjectLabelFactory.from_structured_array(label, (720, 1280), None)
    print(factory)

def test_objectlabelfactory_gen4():
    path = "/datasets/sheusinger/gen4/train/moorea_2019-04-12_000_500000_60500000/labels_v2/labels.npz"
    label = np.load(path)['labels']
    print(label)
    print(label.shape)
    factory = OLF.from_structured_array(label, None, (720, 1280), None)
    print(factory)

def compute_areas(bboxs):
    areas = np.abs(np.multiply(bboxs[:, 2], bboxs[:, 3]))
    return areas


def check_labels(config):
    data_module = fetch_data_module(config=config)
    data_module.setup('fit')
    print(config.dataset.name)
    batch_size = config.batch_size.train
    train_loader = data_module.train_dataloader()
    empty_lables = 0
    for stream_first in tqdm(train_loader):

        data_len = len(stream_first['data'][DataType.EV_REPR])
        obj_labels = list()
        for i in range(data_len):
            frame = stream_first['data'][DataType.EV_REPR][i]
            bbs, indices = stream_first['data'][DataType.OBJLABELS_SEQ][i].get_valid_labels_and_batch_indices()
            if len(bbs) > 0:
                obj_labels.extend(bbs)

        if len(obj_labels) == 0:
            empty_lables += 1
            # for i in range(data_len):
            #     frame = stream_first['data'][DataType.EV_REPR][i]
            #     print(frame.shape)
            #     draw_and_wait(frame, None)

    print(f"Empty labels: {empty_lables}")

def get_grandparent_dir_file(path: str):
    parent_dir = os.path.dirname(path)
    return os.path.dirname(parent_dir)

def do_exploration(config):
    data_module = fetch_data_module(config=config)
    data_module.setup('fit')
    print(config.dataset.name)
    batch_size = config.batch_size.train
    print(batch_size)
    train_loader = data_module.train_dataloader()
    for stream_first in train_loader:
        print(stream_first.keys())
        print(stream_first['data'].keys())
        print(len(stream_first['data'][DataType.EV_REPR]))

        data_len = len(stream_first['data'][DataType.EV_REPR])
        for f in range(batch_size):
            for i in range(data_len):
                frame = stream_first['data'][DataType.EV_REPR][i][f]
                bbs = stream_first['data'][DataType.OBJLABELS_SEQ][i]

                ## This is debugging only. 
                ## Only works if the debug label stuff is there (DataType.EVENT_PATH and LABEL_PATH have to be set)
                # old_bbs, indices = stream_first['data'][DataType.OBJLABELS_SEQ][i].get_valid_labels_and_batch_indices()
                # if (len(old_bbs) == batch_size):
                #     continue
                # event_paths = stream_first['data'][DataType.EVENT_PATH][i]
                # label_paths = stream_first['data'][DataType.LABEL_PATH][i]
                # print(event_paths[f])
                # print(label_paths[f])
                # label_dir = get_grandparent_dir_file(label_paths[f])
                # event_dir = get_grandparent_dir_file(event_paths[f])
                # label_num = label_paths[f].split('_')[-1]
                # event_num = event_paths[f].split('_')[-1]
                # assert label_num == event_num, f"{event_paths[f]}\n{label_paths[f]}\n"
                # assert label_dir == event_dir, f"{event_paths[f]}\n{label_paths[f]}\n"
 
                new_bbs = None
                new_bb = None
                if bbs[f]:
                    new_bb = bbs[f].object_labels[:, 1:5]
                    new_bbs = np.vstack(new_bb)
                draw_and_wait(frame, new_bbs)

def do_exploration_over_batch(config):
    data_module = fetch_data_module(config=config)
    data_module.setup('fit')
    print(config.dataset.name)
    batch_size = config.batch_size.train
    print(batch_size)
    train_loader = data_module.train_dataloader()
    for stream_first in train_loader:
        print(stream_first.keys())
        print(stream_first['data'].keys())
        print(len(stream_first['data'][DataType.EV_REPR]))

        data_len = len(stream_first['data'][DataType.EV_REPR])
        for f in range(batch_size):
            bounding_boxes = list()
            frames = list()
            for i in range(data_len):
                frame = stream_first['data'][DataType.EV_REPR][i][f]
                # bbs, indices = stream_first['data'][DataType.OBJLABELS_SEQ][i].get_valid_labels_and_batch_indices()
                bbs = stream_first['data'][DataType.OBJLABELS_SEQ][i]
                frames.append(frame)
                print(type(bbs))
                print(len(bbs))
                if len(bbs) > 0:
                    if f < len(bbs):
                        new_bb = bbs[f].object_labels[:, 1:5]
                        bounding_boxes.append(new_bb)
            new_bbs = None
            if len(bounding_boxes) > 0:
                new_bbs = np.vstack(bounding_boxes)
            img = torch.vstack(frames)
            print(img.shape)
                
            draw_and_wait(img, new_bbs)

def check_batch_vs_sequence(config):
    data_module = fetch_data_module(config=config)
    data_module.setup('fit')
    print(config.dataset.name)
    batch_size = config.batch_size.train
    print(batch_size)
    train_loader = data_module.train_dataloader()
    for stream_first in train_loader:
        print(stream_first.keys())
        print(stream_first['data'].keys())
        print(len(stream_first['data'][DataType.EV_REPR]))

        data_len = len(stream_first['data'][DataType.EV_REPR])
        for f in range(batch_size):
            bounding_boxes = list()
            frames = list()
            for i in range(data_len):
                frame = stream_first['data'][DataType.EV_REPR][i][f]
                draw_and_display(frame, None, str(i))

            while (key := cv2.waitKey(0)) != ord("r"):
                if (key == ord("q")):
                    cv2.destroyAllWindows()
                    sys.exit(0)

def get_augmented_armasuisse(config: DictConfig):
    arma_dataset = ArmasuisseAugmented.build(DatasetMode.TRAIN, config)
    return arma_dataset

def test_dataset(dataset, batch_size):
    for data in dataset:
        print(len(data))
        data_len = len(data[DataType.EV_REPR])
        for b in range(batch_size):
            labels = data[DataType.OBJLABELS_SEQ][b]
            frame = data[DataType.EV_REPR][b]
            bboxes = None
            if labels:
                bbox = labels.object_labels[:, 1:5]
                bboxes = np.vstack(bbox)
            print(bboxes)
            draw_and_wait(frame.numpy(), bboxes)

def get_dataloader_from_config(config: DictConfig):
    data_module = fetch_data_module(config=config)
    data_module.setup('fit')
    print(f"Loaded dataset: {config.dataset.name}")
    batch_size = config.batch_size.train
    print(f"{batch_size=}")
    train_loader = data_module.train_dataloader()
    return train_loader

def test_dataloader(dataloader, batch_size):
    for data in dataloader:
        print(data.keys())
        print(data['data'].keys())
        print(f"Sequence length: {len(data['data'][DataType.EV_REPR])}")

        data_len = len(data['data'][DataType.EV_REPR])
        for f in range(batch_size):
            for i in range(data_len):
                frame = data['data'][DataType.EV_REPR][i][f].numpy()
                labels = data['data'][DataType.OBJLABELS_SEQ][i]

                bboxes = None
                if labels[f]:
                    bbox = labels[f].object_labels[:, 1:5]
                    bboxes = np.vstack(bbox)
                draw_and_wait(frame, bboxes)

def test_arma_augmented_dataset(config: DictConfig):
    batch_size = config.batch_size.train
    arma_dataloader = get_augmented_armasuisse(config.dataset)
    test_dataset(arma_dataloader, batch_size)

def test_arma_augmented_dataloader(config: DictConfig):
    batch_size = config.batch_size.train
    arma_loader = get_dataloader_from_config(config)
    test_dataloader(arma_loader, batch_size)



@hydra.main(config_path='config', config_name='train', version_base='1.2')
def main(config: DictConfig):
    test_arma_augmented_dataloader(config)
    # test_arma_augmented_dataset(config)
    # do_exploration(config)
    # check_labels(config)
    # check_batch_vs_sequence(config)

if __name__ == '__main__':
    main()
    # test_objectlabelfactory()
    # test_objectlabelfactory_gen4()
