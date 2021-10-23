from dataclasses import asdict, dataclass
from glob import glob
from typing import List

from pandas import DataFrame

from src.checkpoint.checkpoint import Checkpoint
from src.repository.repository import Repository
from src.timer import class_dec_timer


@dataclass(frozen=True)
class DataRepository(Repository):
    bucket_name: str = "kaggle-chaii-2021"
    local_root_path: str = "."

    def __train_df_filepath_from_root(self) -> str:
        return "data/origin/train.csv"

    def __test_df_filepath_from_root(self) -> str:
        return "data/origin/test.csv"

    def __sample_submission_filepath_from_root(self) -> str:
        return "data/origin/sample_submission.csv"

    def __preprocessed_df_filepath_from_root(self, ver: str) -> str:
        return f"data/preprocessed/{ver}.pkl"

    def __checkpoint_filename_from_root(
        self, exp_id: str, fold: int, epoch: int, val_loss: float, val_jaccard: float
    ) -> str:
        return f"data/checkpoint/{exp_id}/{fold}/{epoch}_{val_loss:.4f}_{val_jaccard:.4f}.pkl"

    def __best_checkpoint_filename_from_root(
        self, exp_id: str, fold: int, epoch: int, val_loss: float, val_jaccard: float
    ) -> str:
        return (
            f"data/checkpoint/{exp_id}/best_checkpoint/"
            f"{fold}_{epoch}_{val_loss:.4f}_{val_jaccard:.4f}.pkl"
        )

    def __best_model_state_dict_filename_from_root(
        self, exp_id: str, fold: int, epoch: int, val_loss: float, val_jaccard: float
    ) -> str:
        return (
            f"data/checkpoint/{exp_id}/best_model_state_dict/"
            f"model_state_dict_{fold}_{epoch}_{val_loss:.4f}_{val_jaccard:.4f}.pkl"
        )

    def __gcs_fullpath_to_gcs_filepath(self, gcs_fullpath: str) -> str:
        gcs_filepath = "/".join(gcs_fullpath.split("/")[3:])
        return gcs_filepath

    def __checkpoint_filepath_from_root_to_val_jaccard(
        self, checkpoint_filepath_from_root: str
    ) -> float:
        score = float(
            checkpoint_filepath_from_root.split("/")[-1].split("_")[2][:-4]
        )  # -4 to remove .pkl
        return score

    def load_train_df(self) -> DataFrame:
        filepath_from_root = self.__train_df_filepath_from_root()
        df: DataFrame = self.load(
            filepath_from_root=filepath_from_root,
            mode="dfcsv",
            load_from_gcs=True,
            rm_local_after_load=False,
        )
        return df

    def load_test_df(self) -> DataFrame:
        filepath_from_root = self.__test_df_filepath_from_root()
        df: DataFrame = self.load(
            filepath_from_root=filepath_from_root,
            mode="dfcsv",
            load_from_gcs=True,
            rm_local_after_load=False,
        )
        return df

    def load_sample_submission_df(self) -> DataFrame:
        filepath_from_root = self.__sample_submission_filepath_from_root()
        df: DataFrame = self.load(
            filepath_from_root=filepath_from_root,
            mode="dfcsv",
            load_from_gcs=True,
            rm_local_after_load=False,
        )
        return df

    def preprocessed_df_exists(self, ver: str) -> bool:
        filepath_from_root = self.__preprocessed_df_filepath_from_root(ver=ver)
        gcs_files = self.list_gcs_files(prefix=filepath_from_root)
        if len(gcs_files) == 0:
            return False
        elif len(gcs_files) == 1:
            return True
        else:
            raise Exception("should not occur case.")

    def save_preprocessed_df(self, preprocessed_df: DataFrame, ver: str) -> None:
        filepath_from_root = self.__preprocessed_df_filepath_from_root(ver=ver)
        self.save(
            save_obj=preprocessed_df,
            filepath_from_root=filepath_from_root,
            mode="pkl",
            gcs_mode="cp",
        )

    def load_preprocessed_df(self, ver: str) -> DataFrame:
        filepath_from_root = self.__preprocessed_df_filepath_from_root(ver=ver)
        df = self.load(
            filepath_from_root=filepath_from_root,
            mode="pkl",
            load_from_gcs=True,
            rm_local_after_load=False,
        )
        return df

    def load_checkpoint_from_filepath(
        self, filepath_from_root: str, load_from_gcs: bool, rm_local_after_load: bool
    ) -> Checkpoint:
        """
        to fix the loading format for checkpoint
        """
        checkpoint = Checkpoint(
            **self.load(
                filepath_from_root=filepath_from_root,
                mode="pkl",
                load_from_gcs=load_from_gcs,
                rm_local_after_load=rm_local_after_load,
            )
        )
        return checkpoint

    @class_dec_timer(unit="m")
    def save_checkpoint(self, checkpoint: Checkpoint, is_best: bool) -> None:
        if checkpoint.non_filled_mambers:
            raise Exception(
                f"checkpoint members {checkpoint.non_filled_mambers} are not filled."
            )
        if is_best:
            filepath_from_root = self.__best_checkpoint_filename_from_root(
                exp_id=checkpoint.exp_id,
                fold=checkpoint.fold,
                epoch=checkpoint.epoch,
                val_loss=checkpoint.val_loss,
                val_jaccard=checkpoint.val_jaccard,
            )
            gcs_mode = "mv"
        else:
            filepath_from_root = self.__checkpoint_filename_from_root(
                exp_id=checkpoint.exp_id,
                fold=checkpoint.fold,
                epoch=checkpoint.epoch,
                val_loss=checkpoint.val_loss,
                val_jaccard=checkpoint.val_jaccard,
            )
            gcs_mode = "pass"
        self.save(
            save_obj=asdict(checkpoint),
            filepath_from_root=filepath_from_root,
            mode="pkl",
            gcs_mode=gcs_mode,
            force_save=True,
        )

    def load_checkpoint(self, exp_id: str, fold: int, epoch: int) -> Checkpoint:
        # filepaths = self.list_gcs_files(f"data/checkpoint/{exp_id}/{fold}/{epoch}_")
        filepaths_with_local_root = glob(
            self._filepath_with_local_root(
                f"data/checkpoint/{exp_id}/{fold}/{epoch}_*"
            )
        )
        if len(filepaths_with_local_root) != 1:
            raise Exception(
                f"non-unique fold epoch checkpoint, {filepaths_with_local_root}."
            )
        filepath_with_local_root = filepaths_with_local_root[0]
        checkpoint = self.load_checkpoint_from_filepath(
            filepath_from_root=filepath_with_local_root,
            load_from_gcs=True,
            rm_local_after_load=True,
        )
        return checkpoint

    def best_checkpoint_filepaths(self, exp_id: str) -> List[str]:
        gcs_fullpaths = self.list_gcs_files(
            f"data/checkpoint/{exp_id}/best_checkpoint/"
        )
        gcs_filepaths = [
            self.__gcs_fullpath_to_gcs_filepath(filepath) for filepath in gcs_fullpaths
        ]
        return gcs_filepaths

    def clean_exp_checkpoint(self, exp_id: str) -> None:
        filepaths = self.list_gcs_files(f"data/checkpoint/{exp_id}/")
        for filepath in filepaths:
            prefix = self.__gcs_fullpath_to_gcs_filepath(gcs_fullpath=filepath)
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
            score = self.__checkpoint_filepath_from_root_to_val_jaccard(
                checkpoint_filepath_from_root=filepath
            )
            if score > best_score:
                best_score = score
                best_filepath = filepath
        if best_filepath is None:
            raise Exception("failed to extract best filename.")

        # load best checkpoint and model_state_dict
        # best_prefix = self.__gcs_filepath_to_prefix(gcs_filepath=best_gcs_filepath)
        best_checkpoint = self.load_checkpoint_from_filepath(
            filepath_from_root=best_filepath,
            load_from_gcs=False,
            rm_local_after_load=False,
        )
        self.save_checkpoint(checkpoint=best_checkpoint, is_best=True)

        # model state dict
        best_model_state_dict = best_checkpoint.model_state_dict
        if best_model_state_dict is None:
            raise Exception(f"model weight in {best_filepath} is None.")
        best_model_state_dict_filepath = self.__best_model_state_dict_filename_from_root(
            exp_id=best_checkpoint.exp_id,
            fold=best_checkpoint.fold,
            epoch=best_checkpoint.epoch,
            val_loss=best_checkpoint.val_loss,
            val_jaccard=best_checkpoint.val_jaccard,
        )
        self.save(
            save_obj=best_model_state_dict,
            filepath_from_root=best_model_state_dict_filepath,
            mode="pkl",
            gcs_mode="mv",
            force_save=True,
        )

        # delete checkpoints
        for filepath in filepaths:
            self.delete(
                filepath_from_root=filepath,
                delete_from_local=True,
                delete_from_gcs=False,
            )
