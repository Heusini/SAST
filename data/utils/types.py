from enum import auto, Enum

try:
    from enum import StrEnum
except ImportError:
    from strenum import StrEnum
from typing import Dict, List, Optional, Tuple, Union

import torch as th

from data.utils.object_labels import ObjectLabels
from data.utils.sparsely_batched_object_labels import SparselyBatchedObjectLabels


class DataType(Enum):
    EV_REPR = auto()
    FLOW = auto()
    IMAGE = auto()
    OBJLABELS = auto()
    OBJLABELS_SEQ = auto()
    IS_PADDED_MASK = auto()
    IS_FIRST_SAMPLE = auto()
    TOKEN_MASK = auto()
    EVENT_PATH = auto()
    LABEL_PATH = auto()


class DatasetType(Enum):
    GEN1 = auto()
    GEN4 = auto()
    ARMA = auto()
    ERGB = auto()


class DatasetMode(Enum):
    TRAIN = auto()
    VALIDATION = auto()
    TESTING = auto()


class DatasetSamplingMode(StrEnum):
    RANDOM = 'random'
    STREAM = 'stream'
    MIXED = 'mixed'


class ObjDetOutput(Enum):
    LABELS_PROPH = auto()
    PRED_PROPH = auto()
    EV_REPR = auto()
    SPARSITY_MASK = auto()
    IMAGE_DATA = auto()
    SKIP_VIZ = auto()
    R_L = auto()

class ModelOutput(Enum):
    LOSSES = auto()
    P = auto()
    PREDICTIONS = auto()
    GROUND_TRUTHS = auto()
    EVENT_DATA = auto()
    SPARSITY_MASK = auto()
    IMAGE_DATA = auto()


LoaderDataDictGenX = Dict[DataType, Union[List[th.Tensor], ObjectLabels, SparselyBatchedObjectLabels, List[bool]]]

LstmState = Optional[Tuple[th.Tensor, th.Tensor]]
LstmStates = List[LstmState]

FeatureMap = th.Tensor
BackboneFeatures = Dict[int, th.Tensor]
