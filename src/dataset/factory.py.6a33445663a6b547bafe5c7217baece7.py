from typing import List, Tuple

from pandas import DataFrame
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.dataset.chaii_dataset import ChaiiDataset
from src.factory import Factory
from src.log import myLogger


class DatasetFactory(Factory[Dataset]):
    def __init__(
        self,
        dataset_type: str,
        tokenizer_type: str,
        aug: List[str],
        logger: myLogger,
    ) -> None:
        super().__init__(
            dataset_type=dataset_type,
            tokenizer_type=tokenizer_type,
            aug=aug,
            logger=logger,
        )

    def _create(self, df: DataFrame, is_train: bool) -> Dataset:
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_type)

        if self.dataset_type == "chaii":
            dataset = ChaiiDataset(
                df=df, tokenizer=tokenizer, is_train=is_train, logger=self.logger
            )
        else:
            raise NotImplementedError()
        return dataset
