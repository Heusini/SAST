import pytorch_lightning as pl
from typing import Any, Dict, Optional, Union
from pathlib import Path
import numpy as np
from omegaconf import DictConfig, ListConfig
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from data.utils.types import DatasetMode, DatasetSamplingMode
from data.utils.collate import custom_collate_rnd, custom_collate_streaming
from tqdm import tqdm

from data.general.augmented import AugmentedDataset
from data.general.partial_dataset import PartialDataset
from data.arma_utils.armasuisse import ArmasuisseDataset
from data.base_dataset import BaseDataset



class ArmaDataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset_config: DictConfig,
                 num_workers_train: int,
                 num_workers_eval: int,
                 batch_size_train: int,
                 batch_size_eval: int,
                 base_dataset: BaseDataset):
        super().__init__()
        assert num_workers_train >= 0
        assert num_workers_eval >= 0
        assert batch_size_train >= 1
        assert batch_size_eval >= 1

        self.base_dataset = base_dataset

        self.num_workers_train = num_workers_train
        self.num_workers_eval = num_workers_eval
        self.dataset_config = dataset_config
        self.train_sampling_mode = dataset_config.train.sampling
        self.eval_sampling_mode = dataset_config.eval.sampling

        self.overall_batch_size_train = batch_size_train
        self.overall_batch_size_eval = batch_size_eval
        

        assert self.train_sampling_mode in iter(DatasetSamplingMode)
        assert self.eval_sampling_mode in (DatasetSamplingMode.STREAM, DatasetSamplingMode.RANDOM)

        # this is not implemented yet
        # if self.eval_sampling_mode == DatasetSamplingMode.STREAM:
        #     self.build_eval_dataset = partial(build_streaming_dataset,
        #                                       batch_size=self.overall_batch_size_eval,
        #                                       num_workers=self.overall_num_workers_eval)
        # elif self.eval_sampling_mode == DatasetSamplingMode.RANDOM:
        #     self.build_eval_dataset = build_random_access_dataset
        # else:
        #     raise NotImplementedError

        self.sampling_mode_2_dataset = dict()
        self.sampling_mode_2_train_workers = dict()
        self.sampling_mode_2_train_batch_size = dict()
        self.sampling_mode_2_train_batch_size[DatasetSamplingMode.RANDOM] = batch_size_train
        self.validation_dataset = None
        self.test_dataset = None
    

        

    def setup(self, stage: Optional[str] = None) -> None:
        use_fraction_train = self.dataset_config.train.use_fraction
        use_fraction_val = self.dataset_config.validation.use_fraction
        if stage == 'fit':
            train_datasets = []
            validation_datasets = []
            path = self.dataset_config.path
            if not isinstance(path, (list, ListConfig)):
                path = [path]
            else:
                path = list(path)

            for i, dataset_path in enumerate(path):
                percent_train = use_fraction_train
                percent_val = use_fraction_val
                if isinstance(percent_train, (list, ListConfig)):
                    percent_train = float(percent_train[i])
                if isinstance(percent_val, (list, ListConfig)):
                    percent_val = float(percent_val[i])

                # this is not optimal as we override the datasetpath maybe change
                self.dataset_config.path = str(dataset_path)
                train_dataset = self.base_dataset.build(dataset_mode=DatasetMode.TRAIN, 
                                                              dataset_config=self.dataset_config)
                train_dataset = PartialDataset(train_dataset, percent_train)
                if self.dataset_config.data_augmentation:
                    train_dataset = AugmentedDataset.build(dataset_config=self.dataset_config, dataset=train_dataset)
                train_datasets.append(train_dataset)
            
                validation_dataset = self.base_dataset.build(dataset_mode=DatasetMode.VALIDATION, 
                                                                  dataset_config=self.dataset_config)
                validation_dataset = PartialDataset(validation_dataset, percent_val) 
                validation_datasets.append(validation_dataset)
            self.sampling_mode_2_dataset[DatasetSamplingMode.RANDOM] = ConcatDataset(train_datasets)
            self.validation_dataset = ConcatDataset(validation_datasets)

        elif stage == 'validate':
            self.validation_dataset = self.base_dataset.build(dataset_mode=DatasetMode.VALIDATION,
                                                              dataset_config=self.dataset_config)
        elif stage == 'test':
            print("test")
            raise NotImplementedError
        else: 
            raise NotImplementedError
    def train_dataloader(self):
        dataset = self.sampling_mode_2_dataset[DatasetSamplingMode.RANDOM]
        batch_size = self.sampling_mode_2_train_batch_size[DatasetSamplingMode.RANDOM]
        shuffle = self.dataset_config.train.shuffle
        return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          sampler=None,
                          num_workers=self.num_workers_train,
                          pin_memory=True,
                          drop_last=True,
                          collate_fn=custom_collate_rnd)
    def val_dataloader(self):
        dataset = self.validation_dataset
        batch_size = self.overall_batch_size_eval
        shuffle = self.dataset_config.validation.shuffle
        return DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          sampler=None,
                          num_workers=self.num_workers_eval,
                          pin_memory=True,
                          drop_last=True,
                          collate_fn=custom_collate_rnd)
