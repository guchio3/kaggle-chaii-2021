from typing import List

from pandas import DataFrame
from torch.utils.data import Dataset

from src.dataset.chaii_dataset import ChaiiDataset
from src.factory import Factory
from src.log import myLogger


class DatasetFactory(Factory[Dataset]):
    def __init__(
        self,
        dataset_type: str,
        aug: List[str],
        logger: myLogger,
    ) -> None:
        super().__init__(
            dataset_type=dataset_type,
            aug=aug,
            logger=logger,
        )

    def _create(self, df: DataFrame, is_test: bool) -> Dataset:
        if self.dataset_type == "chaii":
            dataset = ChaiiDataset(
                df=df, aug=self.aug, is_test=is_test, logger=self.logger
            )
        else:
            raise NotImplementedError()
        return dataset
