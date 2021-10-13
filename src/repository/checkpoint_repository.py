from dataclasses import dataclass

from src.checkpoint.checkpoint import Checkpoint
from src.log import myLogger
from src.repository.repository import Repository


@dataclass(frozen=True)
class CheckpointRepository(Repository):
    logger: myLogger
    bucket_name: str = "kaggle-chaii-2021"

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        filepath = f"./data/checkpoints/{checkpoint.exp_id}/{checkpoint.fold}/{checkpoint.epoch}.pkl"
        self.save(
            save_obj=checkpoint,
            filepath=filepath,
            mode="pkl",
            gcs_mode="mv",
            force_save=True,
        )

    def load_checkpoint(self, exp_id: str, fold: int, epoch: int) -> Checkpoint:
        filepath = f"./data/checkpoints/{exp_id}/{fold}/{epoch}.pkl"
        self.load(filepath=filepath, mode="pkl", load_from_gcs=True, rm_after_load=True)

    def extract_and_save_best_fold_epoch(self, exp_id: str, fold: int) -> None:
