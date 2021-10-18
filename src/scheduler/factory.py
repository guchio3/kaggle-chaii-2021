from src.factory import Factory
from src.log import myLogger
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler


class SchedulerFactory(Factory[_LRScheduler]):
    def __init__(
        self,
        scheduler_type: str,
        max_epoch: int,
        cosine_eta_min: float,
        logger: myLogger,
    ):
        super().__init__(
            scheduler_type=scheduler_type,
            max_epoch=max_epoch,
            cosine_eta_min=cosine_eta_min,
            logger=logger,
        )

    def _create(self, optimizer: Optimizer) -> _LRScheduler:
        if self.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.max_epoch - 1,
                eta_min=self.cosine_eta_min,
            )
        else:
            raise NotImplementedError
        return scheduler
