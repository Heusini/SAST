import cv2
import numpy as np
from typing import List

class HeatMap:
    def __init__(self, im_size, colormap):
        self.im_size = im_size
        self.heatmap = np.zeros(im_size)
        self.colormap = colormap

    def index_to_coordinate(self, index):
        row = index // self.im_size[1]
        col = index - row * self.im_size[1]
        return row, col


    def get_max_val(self):
        indices = np.argmax(self.heatmap)
        y, x = self.index_to_coordinate(indices)
        max_value = self.heatmap[y, x]
        return (max_value, y, x)

    def heatmap_effect(self, box: List[float]) -> None:
            """
            Efficiently calculate heatmap area and effect location for applying colormap.

            Args:
                box (List[float]): Bounding box coordinates [x0, y0, x1, y1].
            """
            x0, y0, x1, y1 = map(int, box)
            radius_squared = (min(x1 - x0, y1 - y0) // 2) ** 2

            # Create a meshgrid with region of interest (ROI) for vectorized distance calculations
            xv, yv = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))

            # Calculate squared distances from the center
            dist_squared = (xv - ((x0 + x1) // 2)) ** 2 + (yv - ((y0 + y1) // 2)) ** 2

            # Create a mask of points within the radius
            within_radius = dist_squared <= radius_squared

            # Update only the values within the bounding box in a single vectorized operation
            self.heatmap[y0:y1, x0:x1][within_radius] += 1


    def process(self, boxes: List[np.ndarray]):
        count = 0
        for box in boxes:
            self.heatmap_effect(box)
            count += 1

        normalized_heatmap = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored_heatmap = cv2.applyColorMap(normalized_heatmap, self.colormap)

        return colored_heatmap
