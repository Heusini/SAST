import pytorch_lightning as pl
from typing import Any, Dict, Optional, Union
from pathlib import Path
import numpy as np
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, Dataset, DataLoader
from data.utils.types import DatasetMode, DatasetSamplingMode
from data.utils.collate import custom_collate_rnd, custom_collate_streaming
from tqdm import tqdm

from data.general.augmented import AugmentedDataset
from data.general.partial_dataset import PartialDataset
from data.arma_utils.armasuisse import ArmasuisseDataset



class ArmaDataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset_config: DictConfig,
                 num_workers_train: int,
                 num_workers_eval: int,
                 batch_size_train: int,
                 batch_size_eval: int):
        super().__init__()
        assert num_workers_train >= 0
        assert num_workers_eval >= 0
        assert batch_size_train >= 1
        assert batch_size_eval >= 1

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
        percent_train = self.dataset_config.train.use_fraction
        percent_val = self.dataset_config.validation.use_fraction
        if stage == 'fit':
            train_dataset = ArmasuisseDataset.build(dataset_mode=DatasetMode.TRAIN, 
                                                          dataset_config=self.dataset_config)
            train_dataset = PartialDataset(train_dataset, percent_train)
            if self.dataset_config.data_augmentation:
                train_dataset = AugmentedDataset.build(dataset_config=self.dataset_config, dataset=train_dataset)
            self.sampling_mode_2_dataset[DatasetSamplingMode.RANDOM] = train_dataset
            
            validation_dataset = ArmasuisseDataset.build(dataset_mode=DatasetMode.VALIDATION, 
                                                              dataset_config=self.dataset_config)
            validation_dataset = PartialDataset(validation_dataset, percent_val) 
            self.validation_dataset = validation_dataset

        elif stage == 'validate':
            self.validation_dataset = ArmasuisseDataset.build(dataset_mode=DatasetMode.VALIDATION,
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
                          shuffle=True,
                          sampler=None,
                          num_workers=self.num_workers_eval,
                          pin_memory=True,
                          drop_last=True,
                          collate_fn=custom_collate_rnd)




def build_random_access_dataset_arma(dataset_mode: DatasetMode, dataset_config: DictConfig):
    dataset_path = Path(dataset_config.path)
    assert dataset_path.is_dir(), f'{str(dataset_path)}'
    mode2str = {DatasetMode.TRAIN: 'train',
                DatasetMode.VALIDATION: 'val',
                DatasetMode.TESTING: 'test'}

    dataset = build_arma(mode2str[dataset_mode], dataset_config)

    return dataset
    # for entry in tqdm(split_path.iterdir(), desc=f'creating rnd access {mode2str[dataset_mode]} datasets'):
    #     print(entry)
