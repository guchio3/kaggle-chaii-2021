from torch.optim import Adam, Optimizer

from src.factory import Factory
from src.log import myLogger
from src.model.model import Model


class OptimizerFactory(Factory[Optimizer]):
    def __init__(
        self, optimizer_type: str, learning_rate: float, logger: myLogger,
    ):
        super().__init__(
            optimizer_type=optimizer_type, learning_rate=learning_rate, logger=logger,
        )

    def _create(self, model: Model) -> Optimizer:
        if self.optimizer_type == "adam":
            optimizer = Adam(params=model.parameters(), lr=self.learning_rate)
        else:
            raise NotImplementedError
        return optimizer
