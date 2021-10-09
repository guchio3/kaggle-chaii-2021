from dataclasses import dataclass
from typing import Dict

from src.fold import Fold
from src.repository import Repository


@dataclass
class FoldRepository(Repository):
    bucket_name: str = "kaggle-chaii-2021"

    def save_fold(self, exp_id: str, fold_num: int) -> None:
        1

    def load_fold(self, ) -> Fold:
        1

    def load_folds(self, ) -> Dict[int, Fold]:
        1
