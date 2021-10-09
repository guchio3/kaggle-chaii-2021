from dataclasses import dataclass

import pandas as pd
from pandas import DataFrame

from src.constants import Constants
from src.logs import myLogger


@dataclass(frozen=True)
class ModelRepository:
    exp_id: str
    logger: myLogger
    origin_data_dir: str = Constants.origin_data_dir

    @property
    def _(self) -> DataFrame:
        1
