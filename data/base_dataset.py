from abc import ABC, abstractmethod
from data.utils.types import DatasetMode
from torch.utils.data import ConcatDataset, Dataset
from omegaconf import DictConfig

class BaseDataset(Dataset):
    def __init__(self) -> None: 
        super.__init__()

    @staticmethod
    @abstractmethod
    def build(dataset_mode: DatasetMode, dataset_config: DictConfig):
        pass

