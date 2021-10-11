from dataclasses import dataclass
from typing import OrderedDict

from torch import Tensor


@dataclass
class Checkpoint:
    fold: int
    epoch: int
    model_state_dict: OrderedDict[str, Tensor]
    optimizer_state_dict: OrderedDict[str, Tensor]
    scheduler_state_dict: OrderedDict[str, Tensor]
