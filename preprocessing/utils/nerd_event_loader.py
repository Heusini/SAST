import h5py
import numpy as np
# import cupy as cp
import math


class EventLoader:
    def __init__(self, input_path, ts=None, delta_t=50000):

        self.events = np.load(input_path)['data']
        self.x = self.events[:, 0]
        self.y = self.events[:, 1]
        self.p = self.events[:, 2]
        self.t = self.events[:, 3]

        start_time = self.t[0]
        end_time = self.t[-1]
        event_batch_count = math.ceil((end_time - start_time + 1) / delta_t)
        end_time = event_batch_count * delta_t

        t = self.t[:]
        batch_start_times = np.linspace(start_time, start_time + end_time, num = event_batch_count, endpoint = False)
        batch_end_times = np.linspace(start_time + delta_t, start_time + end_time, num = event_batch_count, endpoint = True)
        self.start_indices = np.searchsorted(t, batch_start_times, side='left')
        self.end_indices = np.searchsorted(t, batch_end_times, side='right')

    # def __del__(self):
    #     self.events.close()

    def __len__(self):
        return len(self.start_indices)

    def get_start_time(self, index):
        assert index >= 0 and index <= len(self)
        return self.t[self.start_indices[index]]

    def get_end_time(self, index):
        assert index >= 0 and index <= len(self)
        # look maybe for a better solution to this. The problem here is that the last
        # index of end_indices is len(self.t) and is out of bounds
        # maybe there is a smater way to solve this but right now i am to tired
        return self.t[:self.end_indices[index]][-1]

    def get_item(self, frame):
        # Check if the frame existing
        if frame < 0 or frame >= len(self.start_indices):
            return False

        # Return the data at a particular index
        batch_event_start_idx = int(self.start_indices[frame])
        batch_event_end_idx = int(self.end_indices[frame])

        # Create structured array
        events = np.empty(batch_event_end_idx - batch_event_start_idx, dtype=[('x', '<u2'), ('y', '<u2'), ('p', '<i4'), ('t', '<i8')])
        events['x'] = self.x[batch_event_start_idx : batch_event_end_idx]
        events['y'] = self.y[batch_event_start_idx : batch_event_end_idx]
        events['p'] = self.p[batch_event_start_idx : batch_event_end_idx] > 0
        events['t'] = self.t[batch_event_start_idx : batch_event_end_idx]

        return events
