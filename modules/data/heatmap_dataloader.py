import os
import time
import cv2
import sys
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt


sys.path.append(".")
from modules.utils.fetch import fetch_data_module, fetch_model_module
from data.utils.types import DataType, LoaderDataDictGenX, DatasetMode
import cv2
import numpy as np
from typing import List
import bbox_visualizer as bbv
from heatmap import HeatMap

@hydra.main(config_path='../../config', config_name='train', version_base='1.2')
def main(config: DictConfig):
    heatmap = HeatMap((360, 640), colormap=cv2.COLORMAP_JET)
    data_module = fetch_data_module(config=config)
    data_module.setup('fit')
    train_loader = data_module.train_dataloader()
    bounding_boxes = []
    start = time.time()
    count = 0
    for batch in train_loader:
        data = batch["data"]
        sequence_len = len(data[DataType.IMAGE])
        sparse_obj_labels = data[DataType.OBJLABELS_SEQ]
        for tidx in range(sequence_len):
            current_labels = [l.object_labels.numpy() for l in sparse_obj_labels[tidx].sparse_object_labels_batch]
            if len(current_labels) > 1:
                current_labels = np.vstack(current_labels)
            bounding_boxes.append(current_labels)
    # print(bounding_boxes)
    # labels = [k.object_labels.numpy() for k in bounding_boxes]
    labels = np.vstack(bounding_boxes)
    labels = labels[:, 1:5]
    labels[:,2:] += labels[:, :2]
    labels = labels.tolist()
    end = time.time()
    print(f"Elapsed time: {end-start:.4f} seconds")
    ht_img_total = heatmap.process(labels)
    max, y, x = heatmap.get_max_val()
    label = str(max)
    y *= 2
    x *= 2

    img = cv2.resize(ht_img_total, (1280, 720), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.annotate(label,
             xy=(x, y),                      # Point to annotate
             xytext=(x, y - 50),        # Text location
             arrowprops=dict(arrowstyle='->', color='black', lw=2),
             bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="white", lw=1),
             color='white',
             fontsize=10)
    plt.show()

if __name__ == "__main__":
    main()
