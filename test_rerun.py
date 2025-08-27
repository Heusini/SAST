import os
import cv2
import numpy as np
import rerun as rr
from rerun import Box2DFormat
from data.event_rgb.event_rgb_dataset import EventRGBDataset
from modules.data.armasuisse import ArmaDataModule 
import hydra
from omegaconf import DictConfig, OmegaConf
from data.utils.types import DataType, LoaderDataDictGenX, DatasetMode


def event_heatmap(event_frame):
    img = event_frame
    img = np.sum(img, axis=0)
    img = cv2.applyColorMap(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
    return img

def convert_image(image):
    img = image
    img = img.squeeze(0).permute(1,2,0).numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = img * 255
    img = img.astype(np.uint8)
    return img


@hydra.main(config_path='config', config_name='train', version_base='1.2')
def main(config: DictConfig):
    OmegaConf.to_container(config, resolve=True, throw_on_missing=False)
    sequence_length = config.dataset.sequence_length
    batch_size = config.batch_size.train

    dataset = EventRGBDataset
    dataloader = ArmaDataModule(config.dataset, 4, 4, batch_size, batch_size, dataset)
    dataloader.setup('fit')
    train_loader = dataloader.train_dataloader()
    rr.init('dataloader')
    rr.connect_grpc("rerun+http://127.0.0.1:9876/proxy")


    batch_count = 0
    max_batch_count = 10
    time = 0
    count_img = 0
    for batch in train_loader:
        data = batch['data']
        events = data.get(DataType.EV_REPR)
        image = data.get(DataType.IMAGE)
        boxes = data.get(DataType.OBJLABELS_SEQ)
        for b in range(batch_size):
            for s in range(sequence_length):
                time = count_img * 0.05
                rr.set_time("stable_time", duration=time)
                eve = event_heatmap(events[s][b].numpy())
                img = image[s][b]
                box = boxes[s][b].object_labels[:,1:5].numpy()
                img = convert_image(img)
                rr.log("BOXES", rr.Boxes2D(array=box, array_format=Box2DFormat.XYWH))
                rr.log("RGB_DATA",rr.Image(img, color_model='BGR'))
                rr.log("Event_DATA",rr.Image(eve, color_model='BGR'))
                count_img += 1
        if batch_count > max_batch_count:
            break
        batch_count+=1

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
