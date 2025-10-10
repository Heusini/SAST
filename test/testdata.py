import numpy as np
from dataclasses import dataclass

@dataclass
class TestData():
    base_path = "test/data/drone"
    img = np.load(f"{base_path}/rgbs/rgb_0.npz")['arr_0']
    label = np.load(f"{base_path}/labels/label_0.npz")['arr_0']
