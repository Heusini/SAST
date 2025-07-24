import numpy as np 
import bbox_visualizer as bbv
from typing import List

class Visualizer:
    def __init__(self, batch_size, frame_processors, bbox_processor, pred_processor):
        self.batch_size = batch_size
        self.frame_processors = frame_processors
        self.bbox_processor = bbox_processor
        self.pred_processor = pred_processor

    def render_image(self, frames:List[np.ndarray],
                     bbox_frame: List[int],
                     labels:np.ndarray,
                     predictions: np.ndarray):

        assert len(frames) == len(bbox_frame) == len(self.frame_processors)
        bboxes = self.bbox_processor(labels)
        preds = self.pred_processor(predictions)
        output_frames = list()
        for batch in range(self.batch_size):
            concat_frame = list()
            for i, (f, bbox_id) in enumerate(zip(frames, bbox_frame)):
                batch_frame = self.frame_processors[i](f[batch])
                if bbox_id:
                    if bboxes[batch] is not None and len(bboxes[batch]) > 0:
                        batch_frame = bbv.draw_multiple_rectangles(batch_frame, bboxes[batch].tolist(), thickness=1)
                    if preds[batch] is not None:
                        batch_frame = bbv.draw_multiple_rectangles(batch_frame, preds[batch].tolist(), thickness=1, bbox_color=(0, 0, 255))
                concat_frame.append(batch_frame)
            single_frame = np.hstack(concat_frame)
            output_frames.append(single_frame)

        return output_frames
