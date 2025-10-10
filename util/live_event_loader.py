import h5py, hdf5plugin
import numpy as np
import cupy as cp
import math

from typing import Optional, Tuple



class Events:
    def __init__(self, input_path, ts=None, delta_t=50000, use_gpu=False):
        self.h5f = h5py.File(input_path, 'r')

        self.p = self.h5f['events/p']
        self.t = self.h5f['events/t']
        self.x = self.h5f['events/x']
        self.y = self.h5f['events/y']
        delta_half = delta_t // 2
        start_times = np.clip(ts[:, 1] - delta_half, self.t[0], self.t[-1]).astype(int)
        end_times = np.clip(ts[:, 1] + delta_half, self.t[0], self.t[-1]).astype(int)

        if use_gpu:
            t_cp = cp.asarray(self.t[:])
            self.start_indices = cp.searchsorted(t_cp, cp.asarray(start_times), side='left')
            self.end_indices = cp.searchsorted(t_cp, cp.asarray(end_times), side='right')
        else:
            self.start_indices = np.searchsorted(self.t, start_times, side='left')
            self.end_indices = np.searchsorted(self.t, end_times, side='right')
    def __del__(self):
        self.h5f.close()


class LiveLoader:
    def __init__(self, live_event_loader, bbox_data):
        self.live_event_loader = live_event_loader
        self.bboxes = np.load(bbox_data)

    def get_targets(self, id: int):
        return self.bboxes[self.bboxes['frame'] == id]
  
  
class LiveEventLoader:
    def __init__(self, input_path, ts=None, delta_t=50000, use_gpu=False):
        self.input_path = input_path
        self.ts = ts
        self.delta_t = delta_t
        self.use_gpu = use_gpu
        self.events = None


    @property
    def size(self) -> int:
        return len(self.ts)

    def get_event_instance(self):
        if not self.events:
            self.events = Events(self.input_path, self.ts, self.delta_t, self.use_gpu)
        return self.events

  
    def get_item(self, frame) -> Tuple[np.array, bool]:
        event = self.get_event_instance()
        # Check if the frame existing
        if frame < 0 or frame >= self.size:
          return None, False
        
        # Return the data at a particular index
        batch_event_start_idx = int(event.start_indices[frame])
        batch_event_end_idx = int(event.end_indices[frame])

        # Create structured array
        events = np.empty(batch_event_end_idx - batch_event_start_idx, dtype=[('x', '<u2'), ('y', '<u2'), ('p', '<i4'), ('t', '<i8')])
        events['x'] = event.x[batch_event_start_idx : batch_event_end_idx]
        events['y'] = event.y[batch_event_start_idx : batch_event_end_idx]
        events['p'] = event.p[batch_event_start_idx : batch_event_end_idx] > 0
        events['t'] = event.t[batch_event_start_idx : batch_event_end_idx]

        return events, True
