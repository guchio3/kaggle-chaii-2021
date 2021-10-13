from dataclasses import asdict, dataclass
from typing import Dict, Tuple

from numpy import ndarray
from pandas import DataFrame
from torch import Tensor

from src.checkpoint.checkpoint import Checkpoint
from src.log import myLogger
from src.repository.repository import Repository


@dataclass(frozen=True)
class DataRepository(Repository):
    logger: myLogger
    bucket_name: str = "kaggle-chaii-2021"

    def load_train_df(self) -> DataFrame:
        filepath = "data/origin/train.csv"
        df: DataFrame = self.load(
            filepath=filepath,
            mode="dfcsv",
            load_from_gcs=True,
            rm_local_after_load=False,
        )
        return df

    def load_test_df(self) -> DataFrame:
        filepath = "data/origin/test.csv"
        df: DataFrame = self.load(
            filepath=filepath,
            mode="dfcsv",
            load_from_gcs=True,
            rm_local_after_load=False,
        )
        return df

    def load_sample_submission_df(self) -> DataFrame:
        filepath = "data/origin/sample_submission.csv"
        df: DataFrame = self.load(
            filepath=filepath,
            mode="dfcsv",
            load_from_gcs=True,
            rm_local_after_load=False,
        )
        return df

    def save_sub_df(self, sub_df: DataFrame, exp_id: str) -> None:
        filepath = f"data/submission/sub_{exp_id}.csv"
        self.save(
            save_obj=sub_df,
            filepath=filepath,
            mode="dfcsv",
            gcs_mode="pass",
            force_save=True,
        )

    def load_sub_df(self, exp_id: str) -> DataFrame:
        filepath = f"data/submission/sub_{exp_id}.csv"
        df = self.load(
            filepath=filepath,
            mode="dfcsv",
            load_from_gcs=False,
            rm_local_after_load=False,
        )
        return df

    def save_preprocessed_df(self, preprocessed_df: DataFrame, ver: str) -> None:
        filepath = f"data/preprocessed/{ver}.csv"
        self.save(
            save_obj=preprocessed_df,
            filepath=filepath,
            mode="dfcsv",
            gcs_mode="cp",
        )

    def load_preprocessed_df(self, ver: str) -> DataFrame:
        filepath = f"data/preprocessed/{ver}.csv"
        df = self.load(
            filepath=filepath,
            mode="dfcsv",
            load_from_gcs=True,
            rm_local_after_load=False,
        )
        return df

    def save_fold_idxes(
        self, exp_id: str, fold: int, fold_idxes: Tuple[ndarray, ndarray]
    ) -> None:
        filepath = f"data/fold/{exp_id}/{fold}.pkl"
        self.save(
            save_obj=fold_idxes,
            filepath=filepath,
            mode="pkl",
            gcs_mode="cp",
            force_save=False,
        )

    def load_fold_idxes(self, exp_id: str, fold: int) -> Tuple[ndarray, ndarray]:
        filepath = f"data/fold/{exp_id}/{fold}.pkl"
        fold_idxes = self.load(
            filepath=filepath, mode="pkl", load_from_gcs=True, rm_local_after_load=False
        )
        return fold_idxes

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        if checkpoint.non_filled_mambers:
            raise Exception(
                f"checkpoint members {checkpoint.non_filled_mambers} are not filled."
            )
        filepath = (
            f"data/checkpoint/{checkpoint.exp_id}/{checkpoint.fold}/"
            f"{checkpoint.epoch}_{checkpoint.val_loss}_{checkpoint.val_jaccard}.pkl"
        )
        self.save(
            save_obj=asdict(checkpoint),
            filepath=filepath,
            mode="json",
            gcs_mode="mv",
            force_save=True,
        )

    def load_checkpoint(self, exp_id: str, fold: int, epoch: int) -> Checkpoint:
        filepaths = self.list_gcs_files(
            f"data/checkpoint/{exp_id}/{fold}/{epoch}_*.pkl"
        )
        if len(filepaths) != 1:
            raise Exception(f"non-unique fold epoch checkpoint, {filepaths}.")
        filepath = filepaths[0]
        checkpoint = self.__load_checkpoint_from_filepath(filepath=filepath)
        return checkpoint

    def __load_checkpoint_from_filepath(self, filepath: str) -> Checkpoint:
        checkpoint = Checkpoint(
            **self.load(
                filepath=filepath,
                mode="json",
                load_from_gcs=True,
                rm_local_after_load=True,
            )
        )
        return checkpoint

    def clean_best_fold_epoch_checkpoint(self, exp_id: str) -> None:
        best_filepaths = self.list_gcs_files(f"data/checkpoint/{exp_id}/best/*.pkl")
        for best_filepath in best_filepaths:
            self.delete(best_filepath, delete_from_gcs=True)

    def extract_and_save_best_fold_epoch_model_state_dict(
        self, exp_id: str, fold: int
    ) -> Dict[str, Tensor]:
        filepaths = self.list_gcs_files(f"data/checkpoint/{exp_id}/{fold}/*.pkl")
        if len(filepaths) == 0:
            raise Exception("no checkpoints for exp_id:{exp_id} fold: {fold}.")
        best_score = -1.0
        best_filepath = None
        for filepath in filepaths:
            score = float(filepath.split("/")[-1].split("_")[2])
            if score > best_score:
                best_score = score
                best_filepath = filepath
        if best_filepath is None:
            raise Exception("failed to extract best filename.")

        best_checkpoint = self.__load_checkpoint_from_filepath(filepath=best_filepath)
        best_model_state_dict = best_checkpoint.model_state_dict
        if best_model_state_dict is None:
            raise Exception(f"model weight in {best_filepath} is None.")

        best_filename = best_filepath.split("/")[-1]
        best_save_filepath = (
            f"data/checkpoint/{exp_id}/best/model_state_dict_{best_filename}"
        )
        self.save(
            save_obj=best_model_state_dict,
            filepath=best_save_filepath,
            mode="pt",
            gcs_mode="cp",
            force_save=True,
        )

        return best_model_state_dict
