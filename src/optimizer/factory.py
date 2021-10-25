from torch.optim import Adam, Optimizer
from transformers import AdamW

from src.factory import Factory
from src.log import myLogger
from src.model.model import Model


class OptimizerFactory(Factory[Optimizer]):
    def __init__(
        self,
        optimizer_type: str,
        learning_rate: float,
        weight_decay: float,
        logger: myLogger,
    ):
        super().__init__(
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            logger=logger,
        )

    def _create(self, model: Model) -> Optimizer:
        if self.optimizer_type == "adam":
            optimizer = Adam(
                params=model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == "adamw":
            optimizer = AdamW(
                params=model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            raise NotImplementedError
        return optimizer
