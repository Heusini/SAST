import h5py, hdf5plugin
import numpy as np
import math
  
  
class EventLoader:
  def __init__(self, input_path, ts, delta_t=50000):
    assert ts is not None

    self.h5f = h5py.File(input_path, 'r')

    self.p = self.h5f['events/p']
    self.t = self.h5f['events/t']
    self.x = self.h5f['events/x']
    self.y = self.h5f['events/y']

    delta_half = delta_t // 2
    start_times = np.clip(ts[:, 1] - delta_half, self.t[0], self.t[-1]).astype(int)
    end_times = np.clip(ts[:, 1] + delta_half, self.t[0], self.t[-1]).astype(int)
    self.start_indices = np.searchsorted(self.t, start_times, side='left')
    self.end_indices = np.searchsorted(self.t, end_times, side='right')
    
    def __del__(self):
        self.h5f.close()
  
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
