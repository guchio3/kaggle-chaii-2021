from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Fold:
    fold_num: int
    trn_files: List[str]
    val_files: List[str]
    y_true: Optional[List[float]]
    y_pred: Optional[List[float]]
