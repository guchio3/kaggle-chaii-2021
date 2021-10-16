from dataclasses import asdict, dataclass
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
    model_state_dict: Optional[Dict[str, Tensor]] = None
    optimizer_state_dict: Optional[Dict[str, Tensor]] = None
    scheduler_state_dict: Optional[Dict[str, Tensor]] = None
    val_ids: List[str] = []
    val_start_logits: List[float] = []
    val_end_logits: List[float] = []
    val_segmentation_logits: List[float] = []
    val_loss: Optional[float] = None
    val_jaccard: Optional[float] = None

    @property
    def non_filled_mambers(self) -> List[str]:
        non_filled_members = []
        for field, field_value in asdict(self).items():
            if not field_value:
                non_filled_members.append(field)
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

    def extend_str_list_val_info(self, key: str, val_info: Optional[List[str]]) -> None:
        if val_info is not None:
            getattr(self, key).extend(val_info)

    def extend_tensor_val_info(self, key: str, val_info: Optional[Tensor]) -> None:
        if val_info is not None:
            val_info.to("cpu")
            getattr(self, key).extend(val_info.tolist())
