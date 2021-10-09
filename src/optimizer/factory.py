import re

from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from src.factory import Factory
from src.log import myLogger


class OptimizerFactory(Factory[Optimizer]):
    def __init__(
        self,
        optimizer_type: str,
        learning_rate: float,
        logger: myLogger,
    ):
        super().__init__(
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            logger=logger,
        )

    def _create(
        self,
        scheduler: LearningRateSchedule
    ) -> Optimizer:
        if re.match("adam", self.optimizer_type):
            optim = Adam(learning_rate=scheduler)
        else:
            raise NotImplementedError
        return optim
