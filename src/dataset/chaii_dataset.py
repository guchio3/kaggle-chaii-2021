import torch
from pandas import DataFrame
from torch.utils.data import Dataset

from src.log import myLogger


class ChaiiDataset(Dataset):
    def __init__(
        self,
        df: DataFrame,
        logger: myLogger,
    ) -> None:
        self.df = df
        self.logger = logger

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        res = {
            "input_ids": torch.tensor(row["input_ids"]),
            "attention_mask": torch.tensor(row["attention_mask"]),
            "start_position": torch.tensor(row["start_position"]),
            "end_position": torch.tensor(row["end_position"]),
        }
        return res
