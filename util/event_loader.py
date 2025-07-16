import h5py, hdf5plugin
import numpy as np
import cupy as cp
import math

from typing import Optional, Tuple
  
  
class EventLoader:
  def __init__(self, input_path, ts=None, delta_t=50000, use_gpu=False):

    self.h5f = h5py.File(input_path, 'r')

    self.p = self.h5f['events/p']
    self.t = self.h5f['events/t']
    self.x = self.h5f['events/x']
    self.y = self.h5f['events/y']

    if ts is not None:
      # Armasuisse dataset provides a matching from timestamp to frame, apply here
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

    else:
      # When we use 1mpx dataset, we do not have a matching file, so the timestamp 
      # of the frame is just delta_t * frame
      event_batch_count = math.ceil((self.t[-1] + 1) / delta_t)
      end_time = event_batch_count * delta_t

      if use_gpu:
        batch_start_times = cp.linspace(0, end_time, num = event_batch_count, endpoint = False)
        batch_end_times = cp.linspace(delta_t, end_time, num = event_batch_count, endpoint = False)
        t_cp = cp.asarray(self.t[:])
        self.start_indices = cp.asnumpy(cp.searchsorted(t_cp, batch_start_times, side='left'))
        self.end_indices = cp.asnumpy(cp.searchsorted(t_cp, batch_end_times, side='right'))
      else:
        t = self.t[:]
        batch_start_times = np.linspace(0, end_time, num = event_batch_count, endpoint = False)
        batch_end_times = np.linspace(delta_t, end_time, num = event_batch_count, endpoint = True)
        self.start_indices = np.searchsorted(t, batch_start_times, side='left')
        self.end_indices = np.searchsorted(t, batch_end_times, side='right')
    
    def __del__(self):
        self.h5f.close()
  
  def get_item(self, frame) -> Tuple[np.array, bool]:
    # Check if the frame existing
    if frame < 0 or frame >= len(self.start_indices):
      return None, False
    
    # Return the data at a particular index
    batch_event_start_idx = int(self.start_indices[frame])
    batch_event_end_idx = int(self.end_indices[frame])

    # Create structured array
    events = np.empty(batch_event_end_idx - batch_event_start_idx, dtype=[('x', '<u2'), ('y', '<u2'), ('p', '<i4'), ('t', '<i8')])
    events['x'] = self.x[batch_event_start_idx : batch_event_end_idx]
    events['y'] = self.y[batch_event_start_idx : batch_event_end_idx]
    events['p'] = self.p[batch_event_start_idx : batch_event_end_idx] > 0
    events['t'] = self.t[batch_event_start_idx : batch_event_end_idx]

    return events, True
