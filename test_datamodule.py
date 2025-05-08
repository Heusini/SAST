from modules.utils.fetch import fetch_data_module
import cv2
import hydra
import sys
import numpy as np
from omegaconf import DictConfig, OmegaConf
import bbox_visualizer as bbv

from data.utils.types import DataType
from data.arma_utils.labels import ObjectLabelFactory, SparselyBatchedObjectLabels
from data.genx_utils.labels import ObjectLabelFactory as OLF

def draw_and_display(img_data, new_bb):
    print(img_data.shape)
    img = np.sum(img_data.numpy(), axis=0)
    img = np.sum(img, axis=0)
    print(img.shape)
    img = img / np.max(img)
    img = img * 255
    img = np.array(img, np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    if new_bb is not None:
        new_bb = new_bb[:,1:]
        new_bb[:, 2:] += new_bb[:, :2]
        new_bb = new_bb.astype(np.int32)
        img = bbv.draw_multiple_rectangles(img, new_bb.tolist(), thickness=1)

    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("window", img)
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


@hydra.main(config_path='config', config_name='train', version_base='1.2')
def main(config: DictConfig):
    data_module = fetch_data_module(config=config)
    data_module.setup('fit')
    print(config.dataset.name)
    if (config.dataset.name == 'gen4'):
        train_loader = data_module.train_dataloader()

        count = 0
        for stream_first in train_loader:
            # print(stream_first.keys())
            # print(stream_first['data'].keys())
            # print(stream_first['data'][DataType.EV_REPR][0].shape)
            print(type(stream_first['data'][DataType.OBJLABELS_SEQ][1]))
            # print(stream_first['data'][DataType.IS_FIRST_SAMPLE])
            # print(stream_first['data'][DataType.IS_PADDED_MASK])
            if True in stream_first['data'][DataType.IS_FIRST_SAMPLE]:
                print(count)
                count = 0
                input()
            count += 1
        # count = -1
        # for i in stream_first['data'][DataType.OBJLABELS_SEQ]:
        #     count += 1
        #     print(count)
        #     labels, valid_indices = i.get_valid_labels_and_batch_indices()
        #     print(f"valid: {valid_indices}")
        #     for label in labels:
        #        print(f"tensor: {label.get_labels_as_tensors().shape}")
        #        print(label.get_labels_as_tensors())
    elif config.dataset.name == 'arma':
        train_loader = data_module.train_dataloader()
        for stream_first in train_loader:
            print(stream_first.keys())
            print(stream_first['data'].keys())
            print(len(stream_first['data'][DataType.EV_REPR]))

            data_len = len(stream_first['data'][DataType.EV_REPR])
            for i in range(data_len):
                frame = stream_first['data'][DataType.EV_REPR][i]
                bbs = stream_first['data'][DataType.OBJLABELS_SEQ][i]
                bounding_boxes = list()
                for bb in bbs:
                    if bb:
                        bounding_boxes.append(bb.object_labels)
                if len(bounding_boxes) > 0:
                    new_bbs = np.vstack(bounding_boxes)
                else:
                    new_bbs = None
                # print(new_bbs)
                # print(new_bbs.shape)
                draw_and_display(frame, new_bbs)
            # print(stream_first['data'][DataType.IS_FIRST_SAMPLE])
            # print(stream_first['data'][DataType.IS_PADDED_MASK])

if __name__ == '__main__':
    main()
    # test_objectlabelfactory()
    # test_objectlabelfactory_gen4()
