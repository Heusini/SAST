import pytorch_lightning as pl
from omegaconf import DictConfig

from modules.data.genx import DataModule as genx_data_module

from modules.detection import Module
from models.detection.recurrent_backbone.sast_rnn import RNNDetector

from modules.data.armasuisse import ArmaDataModule as genarma_data_module
from data.arma_utils.armasuisse import ArmasuisseDataset
from data.event_rgb.event_rgb_dataset import EventRGBDataset

from modules.event_rgb_step import step as event_rgb_step

from models.detection.event_rgb.detector import EventRGBDetector
from models.detection.yolox_extension.models.detector import YoloXDetector


def fetch_model_module(config: DictConfig) -> pl.LightningModule:
    if config.model.name == 'eventrgb':
        return Module(config, EventRGBDetector, event_rgb_step)
    raise NotImplementedError


def fetch_data_module(config: DictConfig) -> pl.LightningDataModule:
    batch_size_train = config.batch_size.train
    batch_size_eval = config.batch_size.eval
    num_workers_generic = config.hardware.get('num_workers', None)
    num_workers_train = config.hardware.num_workers.get('train', num_workers_generic)
    num_workers_eval = config.hardware.num_workers.get('eval', num_workers_generic)
    dataset_str = config.dataset.name

    dataset = None
    if dataset_str in {'arma'}:
        dataset = ArmasuisseDataset
    if dataset_str in {'eventrgb'}:
        dataset = EventRGBDataset
    if not dataset:
        raise NotImplementedError

    return genarma_data_module(config.dataset,
                            num_workers_train=num_workers_train,
                            num_workers_eval=num_workers_eval,
                            batch_size_train=batch_size_train,
                            batch_size_eval=batch_size_eval,
                            base_dataset=dataset,)



def fetch_backbone_module(config: DictConfig) -> pl.LightningModule:
    model_str = config.model.name
    if model_str == 'rnndet':
        return RNNDetector(config)
    raise NotImplementedError
