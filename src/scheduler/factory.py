import re

from tensorflow.keras.optimizers.schedules import LearningRateSchedule

from src.factory import Factory
from src.log import myLogger
from src.scheduler.cosine_decay import CosineDecay


class SchedulerFactory(Factory[LearningRateSchedule]):
    def __init__(
        self,
        scheduler_type: str,
        initial_learning_rate: float,
        alpha: float,
        logger: myLogger,
    ):
        super().__init__(
            scheduler_type=scheduler_type,
            initial_learning_rate=initial_learning_rate,
            alpha=alpha,
            logger=logger,
        )

    def _create(self, num_all_steps: int) -> LearningRateSchedule:
        if re.match("cosine", self.scheduler_type):
            scheduler = CosineDecay(
                initial_learning_rate=self.initial_learning_rate,
                decay_steps=num_all_steps,
                alpha=self.alpha,
            )
        else:
            raise NotImplementedError
        return scheduler
