import re

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.nn.modules.loss import _Loss

from src.factory import Factory
from src.log import myLogger


class FobjFactory(Factory[_Loss]):
    def __init__(
        self, fobj_type: str, logger: myLogger,
    ):
        super().__init__(
            fobj_type=fobj_type, logger=logger,
        )

    def _create(self,) -> _Loss:
        if self.fobj_type == "bce":
            loss = BCEWithLogitsLoss()
        elif self.fobj_type == "ce":
            loss = CrossEntropyLoss()
        else:
            raise NotImplementedError
        return loss
