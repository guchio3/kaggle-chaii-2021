from dataclasses import asdict, dataclass
from glob import glob
from typing import Tuple

from numpy import ndarray
from pandas import DataFrame

from src.checkpoint.checkpoint import Checkpoint
from src.repository.repository import Repository
from src.timer import class_dec_timer


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
        return f"data/checkpoint/{exp_id}/{fold}/{epoch}_{val_loss:.4f}_{val_jaccard:.4f}.pkl"

    def __gcs_filepath_to_prefix(self, gcs_filepath: str) -> str:
        prefix = "/".join(gcs_filepath.split("/")[3:])
        return prefix

    def __checkpoint_filepath_to_val_jaccard(self, checkpoint_filepath: str) -> float:
        score = float(
            checkpoint_filepath.split("/")[-1].split("_")[2][:-4]
        )  # -4 to remove .pkl
        return score

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
                mode="pkl",
                load_from_gcs=False,
                rm_local_after_load=False,
            )
        )
        return checkpoint

    @class_dec_timer
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
            mode="pkl",
            gcs_mode="pass",
            force_save=True,
        )

    def load_checkpoint(self, exp_id: str, fold: int, epoch: int) -> Checkpoint:
        # filepaths = self.list_gcs_files(f"data/checkpoint/{exp_id}/{fold}/{epoch}_")
        filepaths = glob(f"data/checkpoint/{exp_id}/{fold}/{epoch}_*")
        if len(filepaths) != 1:
            raise Exception(f"non-unique fold epoch checkpoint, {filepaths}.")
        filepath = filepaths[0]
        checkpoint = self.__load_checkpoint_from_filepath(filepath=filepath)
        return checkpoint

    def clean_exp_checkpoint(self, exp_id: str) -> None:
        filepaths = self.list_gcs_files(f"data/checkpoint/{exp_id}/")
        for filepath in filepaths:
            prefix = self.__gcs_filepath_to_prefix(gcs_filepath=filepath)
            self.delete(prefix, delete_from_local=True, delete_from_gcs=True)

    def extract_and_save_best_fold_epoch_model_state_dict(
        self, exp_id: str, fold: int
    ) -> None:
        # gcs_filepaths = self.list_gcs_files(f"data/checkpoint/{exp_id}/{fold}/")
        filepaths = glob(f"data/checkpoint/{exp_id}/{fold}/*")
        if len(filepaths) == 0:
            raise Exception("no checkpoints for exp_id:{exp_id} fold: {fold}.")
        best_score = -1.0
        best_filepath = None
        for filepath in filepaths:
            score = self.__checkpoint_filepath_to_val_jaccard(
                checkpoint_filepath=filepath
            )
            if score > best_score:
                best_score = score
                best_filepath = filepath
        if best_filepath is None:
            raise Exception("failed to extract best filename.")

        # load best checkpoint and model_state_dict
        # best_prefix = self.__gcs_filepath_to_prefix(gcs_filepath=best_gcs_filepath)
        best_checkpoint = self.__load_checkpoint_from_filepath(filepath=best_filepath)
        best_model_state_dict = best_checkpoint.model_state_dict
        if best_model_state_dict is None:
            raise Exception(f"model weight in {best_filepath} is None.")
        # self.delete(best_prefix, delete_from_local=True, delete_from_gcs=False)

        # save results
        best_filename = best_filepath.split("/")[-1]
        best_checkpoint_filepath = (
            f"data/checkpoint/{exp_id}/best_checkpoint/{best_filename}"
        )
        self.save(
            save_obj=best_checkpoint,
            filepath=best_checkpoint_filepath,
            mode="pkl",
            gcs_mode="mv",
            force_save=True,
        )
        best_model_state_dict_filepath = f"data/checkpoint/{exp_id}/best_model_state_dict/model_state_dict_{best_filename}"
        self.save(
            save_obj=best_checkpoint,
            filepath=best_model_state_dict_filepath,
            mode="pkl",
            gcs_mode="mv",
            force_save=True,
        )

        # delete checkpoints
        for filepath in filepaths:
            prefix = self.__gcs_filepath_to_prefix(gcs_filepath=filepath)
            self.delete(prefix, delete_from_local=True, delete_from_gcs=False)
