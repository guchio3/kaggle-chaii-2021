from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional

from src.logs import myLogger
from src.models.factory import ModelFactory
from src.models.repository import ModelRepository


class Model(metaclass=ABCMeta):
    def __init__(
        self, exp_id: str, device: str, model_type: str, logger: myLogger
    ) -> None:
        self.model_repository = ModelRepository(
            exp_id=exp_id,
        )
        self.model = ModelFactory(logger=logger).create(model_type)

    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def pred(self, features: Optional[List[List[Any]]]) -> None:
        raise NotImplementedError()
