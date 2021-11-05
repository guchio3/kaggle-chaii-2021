import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from transformers import get_linear_schedule_with_warmup

from src.factory import Factory
from src.log import myLogger


class SchedulerFactory(Factory[_LRScheduler]):
    def __init__(
        self,
        scheduler_type: str,
        max_epoch: int,
        cosine_eta_min: float,
        warmup_ratio: float,
        logger: myLogger,
    ):
        super().__init__(
            scheduler_type=scheduler_type,
            max_epoch=max_epoch,
            cosine_eta_min=cosine_eta_min,
            warmup_ratio=warmup_ratio,
            logger=logger,
        )

    def _create(
        self,
        optimizer: Optimizer,
        num_epochs: int,
        loader_size: int,
        accum_mod: int,
        schedule_per_batch: bool,
    ) -> _LRScheduler:
        if schedule_per_batch:
            num_scheduling_steps = math.ceil(loader_size / accum_mod) * num_epochs
        else:
            num_scheduling_steps = num_epochs
        if self.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer=optimizer,
                T_max=num_scheduling_steps - 1,
                eta_min=self.cosine_eta_min,
            )
        elif self.scheduler_type == "linear_warmup":
            if self.warmup_ratio > 0:
                num_warmup_steps = int(self.warmup_ratio * num_scheduling_steps)
            else:
                num_warmup_steps = 0
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_scheduling_steps,
            )
        else:
            raise NotImplementedError
        return scheduler
