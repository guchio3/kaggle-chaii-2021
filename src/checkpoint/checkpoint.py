from dataclasses import dataclass
from typing import Dict, List, Optional

from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.model.model import Model


@dataclass
class Checkpoint:
    exp_id: str
    fold: int
    epoch: int
    val_loss: Optional[float]
    best_val_jaccard: Optional[float]
    model_state_dict: Optional[Dict[str, Tensor]] = None
    optimizer_state_dict: Optional[Dict[str, Tensor]] = None
    scheduler_state_dict: Optional[Dict[str, Tensor]] = None
    val_ids: List[str] = []
    val_preds: List[float] = []

    @property
    def non_filled_mambers(self) -> List[str]:
        non_filled_members = []
        if self.val_loss is None:
            non_filled_members.append("val_loss")
        if self.best_val_jaccard is None:
            non_filled_members.append("best_val_jaccard")
        if self.model_state_dict is None:
            non_filled_members.append("model_state_dict")
        if self.optimizer_state_dict is None:
            non_filled_members.append("optimizer_state_dict")
        if self.optimizer_state_dict is None:
            non_filled_members.append("optimizer_state_dict")
        if self.scheduler_state_dict is None:
            non_filled_members.append("scheduler_state_dict")
        return non_filled_members

    def set_model(self, model: Model) -> None:
        model.to("cpu")
        self.model_state_dict = model.state_dict()

    def set_optimizer(self, optimizer: Optimizer) -> None:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor):
                    state[k] = v.to("cpu")
        self.optimizer_state_dict = optimizer.state_dict()

    def set_scheduler(self, scheduler: _LRScheduler) -> None:
        self.scheduler_state_dict = scheduler.state_dict()

    def extend_val_ids(self, val_ids: List[str]) -> None:
        self.val_ids.extend(val_ids)

    def extend_val_preds(self, val_preds: List[float]) -> None:
        self.val_preds.extend(val_preds)
