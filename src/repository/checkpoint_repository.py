from dataclasses import dataclass

from src.log import myLogger
from src.repository.repository import Repository


@dataclass(frozen=True)
class CheckpointRepository(Repository):
    logger: myLogger
    bucket_name: str = "kaggle-chaii-2021"

    def (self, ) -> None:
