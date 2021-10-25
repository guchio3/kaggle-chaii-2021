from typing import Any, Dict, List

import torch
from pandas import DataFrame
from torch.utils.data import Dataset

from src.log import myLogger


class ChaiiDataset(Dataset):
    def __init__(self, df: DataFrame, aug: List[str], logger: myLogger,) -> None:
        self.df = df
        self.aug = aug
        self.logger = logger
        self.mode = "train"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        # res = {
        #     "id": str(row["id"]),
        #     "context": str(row["context"]),
        #     "question": str(row["question"]),
        #     "answer_text": str(row["answer_text"]),
        #     "language": str(row["language"]),
        #     "input_ids": torch.tensor(row["input_ids"]),
        #     "attention_mask": torch.tensor(row["attention_mask"]),
        #     "offset_mapping": torch.tensor(row["offset_mapping"]),
        #     "start_position": torch.tensor(row["start_position"]),
        #     "end_position": torch.tensor(row["end_position"]),
        #     "segmentation_position": torch.tensor(row["segmentation_position"]),
        # }
        # return res
        if self.mode == "train":
            res = {
                # "id": str(row["id"]),
                # "context": str(row["context"]),
                # "question": str(row["question"]),
                # "answer_text": str(row["answer_text"]),
                # "language": str(row["language"]),
                "input_ids": torch.tensor(row["input_ids"]),
                "attention_mask": torch.tensor(row["attention_mask"]),
                # "offset_mapping": torch.tensor(row["offset_mapping"]),
                "start_positions": torch.tensor(row["start_position"]),
                "end_positions": torch.tensor(row["end_position"]),
                # "segmentation_position": torch.tensor(row["segmentation_position"]),
            }
        else:
            res = {
                "input_ids": torch.tensor(row["input_ids"]),
                "attention_mask": torch.tensor(row["attention_mask"]),

                    }
        return res
