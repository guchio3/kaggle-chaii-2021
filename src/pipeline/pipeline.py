from __future__ import annotations

from abc import ABCMeta, abstractmethod

from src.log import myLogger


class Pipeline(metaclass=ABCMeta):
    def __init__(self, pipeline_type: str, exp_id: str, logger: myLogger) -> None:
        self.pipeline_type = pipeline_type
        self.exp_id = exp_id
        self.logger = logger

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError()
