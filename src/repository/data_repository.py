from dataclasses import asdict, dataclass
from typing import Tuple

from numpy import ndarray
from pandas import DataFrame

from src.checkpoint.checkpoint import Checkpoint
from src.repository.repository import Repository


@dataclass(frozen=True)
class DataRepository(Repository):
    bucket_name: str = "kaggle-chaii-2021"

    def __train_df_filepath(self) -> str:
        return "data/origin/train.csv"

    def __test_df_filepath(self) -> str:
        return "data/origin/test.csv"

    def __sample_submission_filepath(self) -> str:
        return "data/origin/sample_submission.csv"

    def __preprocessed_df_filepath(self, ver: str) -> str:
        return f"data/preprocessed/{ver}.pkl"

    def __fold_idxes_filepath(self, exp_id: str, fold: int) -> str:
        return f"data/fold/{exp_id}/{fold}.pkl"

    def __checkpoint_filename(
        self, exp_id: str, fold: int, epoch: int, val_loss: float, val_jaccard: float
    ) -> str:
        return f"data/checkpoint/{exp_id}/{fold}/{epoch}_{val_loss}_{val_jaccard}.pkl"

    def load_train_df(self) -> DataFrame:
        filepath = self.__train_df_filepath()
        df: DataFrame = self.load(
            filepath=filepath,
            mode="dfcsv",
            load_from_gcs=True,
            rm_local_after_load=False,
        )
        return df

    def load_test_df(self) -> DataFrame:
        filepath = self.__test_df_filepath()
        df: DataFrame = self.load(
            filepath=filepath,
            mode="dfcsv",
            load_from_gcs=True,
            rm_local_after_load=False,
        )
        return df

    def load_sample_submission_df(self) -> DataFrame:
        filepath = self.__sample_submission_filepath()
        df: DataFrame = self.load(
            filepath=filepath,
            mode="dfcsv",
            load_from_gcs=True,
            rm_local_after_load=False,
        )
        return df

    def __sub_df_filepath(self, exp_id: str) -> str:
        return f"data/submission/sub_{exp_id}.csv"

    def save_sub_df(self, sub_df: DataFrame, exp_id: str) -> None:
        filepath = self.__sub_df_filepath(exp_id=exp_id)
        self.save(
            save_obj=sub_df,
            filepath=filepath,
            mode="dfcsv",
            gcs_mode="pass",
            force_save=True,
        )

    def load_sub_df(self, exp_id: str) -> DataFrame:
        filepath = self.__sub_df_filepath(exp_id=exp_id)
        df = self.load(
            filepath=filepath,
            mode="dfcsv",
            load_from_gcs=False,
            rm_local_after_load=False,
        )
        return df

    def preprocessed_df_exists(self, ver: str) -> bool:
        filepath = self.__preprocessed_df_filepath(ver=ver)
        gcs_files = self.list_gcs_files(prefix=filepath)
        if len(gcs_files) == 0:
            return False
        elif len(gcs_files) == 1:
            return True
        else:
            raise Exception("should not occur case.")

    def save_preprocessed_df(self, preprocessed_df: DataFrame, ver: str) -> None:
        filepath = self.__preprocessed_df_filepath(ver=ver)
        self.save(
            save_obj=preprocessed_df, filepath=filepath, mode="pkl", gcs_mode="cp",
        )

    def load_preprocessed_df(self, ver: str) -> DataFrame:
        filepath = self.__preprocessed_df_filepath(ver=ver)
        df = self.load(
            filepath=filepath,
            mode="pkl",
            load_from_gcs=True,
            rm_local_after_load=False,
        )
        return df

    def save_fold_idxes(
        self, exp_id: str, fold: int, fold_idxes: Tuple[ndarray, ndarray]
    ) -> None:
        filepath = self.__fold_idxes_filepath(exp_id=exp_id, fold=fold)
        self.save(
            save_obj=fold_idxes,
            filepath=filepath,
            mode="pkl",
            gcs_mode="cp",
            force_save=False,
        )

    def load_fold_idxes(self, exp_id: str, fold: int) -> Tuple[ndarray, ndarray]:
        filepath = self.__fold_idxes_filepath(exp_id=exp_id, fold=fold)
        fold_idxes = self.load(
            filepath=filepath, mode="pkl", load_from_gcs=True, rm_local_after_load=False
        )
        return fold_idxes

    def __load_checkpoint_from_filepath(self, filepath: str) -> Checkpoint:
        """
        to fix the loading format for checkpoint
        """
        checkpoint = Checkpoint(
            **self.load(
                filepath=filepath,
                mode="json",
                load_from_gcs=True,
                rm_local_after_load=True,
            )
        )
        return checkpoint

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        if checkpoint.non_filled_mambers:
            raise Exception(
                f"checkpoint members {checkpoint.non_filled_mambers} are not filled."
            )
        filepath = self.__checkpoint_filename(
            exp_id=checkpoint.exp_id,
            fold=checkpoint.fold,
            epoch=checkpoint.epoch,
            val_loss=checkpoint.val_loss,
            val_jaccard=checkpoint.val_jaccard,
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

    def clean_best_fold_epoch_checkpoint(self, exp_id: str) -> None:
        best_filepaths = self.list_gcs_files(f"data/checkpoint/{exp_id}/best/*.pkl")
        for best_filepath in best_filepaths:
            self.delete(best_filepath, delete_from_gcs=True)

    def extract_and_save_best_fold_epoch_model_state_dict(
        self, exp_id: str, fold: int
    ) -> None:
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
