import re

from tensorflow.keras.losses import BinaryCrossentropy, Loss

from src.factory import Factory
from src.log import myLogger


class FobjFactory(Factory[Loss]):
    def __init__(
        self,
        fobj_type: str,
        logger: myLogger,
    ):
        super().__init__(
            fobj_type=fobj_type,
            logger=logger,
        )

    def _create(
        self,
    ) -> Loss:
        if re.match("BCE", self.fobj_type):
            loss = BinaryCrossentropy()
            # loss = BinaryCrossentropy(from_logits=False, label_smoothing=0, axis=-1)
        else:
            raise NotImplementedError
        return loss
