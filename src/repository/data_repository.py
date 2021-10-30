from dataclasses import asdict, dataclass
from glob import glob
from typing import Dict, Generator, List

from pandas import DataFrame
from torch import Tensor

from src.checkpoint.checkpoint import Checkpoint
from src.repository.repository import Repository
from src.timer import class_dec_timer


@dataclass(frozen=False)
class DataRepository(Repository):
    bucket_name: str = "kaggle-chaii-2021"
    local_root_path: str = "."
    origin_root_path: str = "data/origin"
    dataset_root_path: str = "data/dataset"
    preprocessed_root_path: str = "data/preprocessed"
    checkpoint_root_path: str = "data/checkpoint"

    def __train_df_filepath_from_root(self) -> str:
        return f"{self.origin_root_path}/train.csv"

    def __test_df_filepath_from_root(self) -> str:
        return f"{self.origin_root_path}/test.csv"

    def __sample_submission_filepath_from_root(self) -> str:
        return f"{self.origin_root_path}/sample_submission.csv"

    def __cleaned_train_df_filepath_from_root(self) -> str:
        return f"{self.dataset_root_path}/cleaned-data-for-chaii/cleaned_train.csv"

    def __filepath_to_filename_wo_extension(self, filepath: str) -> str:
        filename = filepath.split("/")[-1]
        filename_wo_extension = filename.split(".")[0]
        return filename_wo_extension

    def __preprocessed_df_filepath_from_root(
        self,
        dataset_name: str,
        class_name: str,
        tokenizer_name: str,
        max_length: int,
        pad_on_right: bool,
        stride: int,
        use_language_as_question: bool,
    ) -> str:
        return (
            f"{self.preprocessed_root_path}/{dataset_name}_{class_name}_"
            f"{tokenizer_name}_{max_length}_{pad_on_right}_"
            f"{stride}_{use_language_as_question}.pkl"
        )

    def __checkpoint_filename_from_root(
        self, exp_id: str, fold: int, epoch: int, val_loss: float, val_jaccard: float
    ) -> str:
        return (
            f"{self.checkpoint_root_path}/{exp_id}/{fold}/"
            f"{epoch}_{val_loss:.4f}_{val_jaccard:.4f}.pkl"
        )

    def __best_checkpoint_filename_from_root(
        self, exp_id: str, fold: int, epoch: int, val_loss: float, val_jaccard: float
    ) -> str:
        return (
            f"{self.checkpoint_root_path}/{exp_id}/best_checkpoint/"
            f"{fold}_{epoch}_{val_loss:.4f}_{val_jaccard:.4f}.pkl"
        )

    def __best_model_state_dict_directory_from_root(self, exp_id: str) -> str:
        return f"{self.checkpoint_root_path}/{exp_id}/best_model_state_dict"

    def __best_kaggle_kernel_model_state_dict_directory_from_root(
        self, exp_id: str
    ) -> str:
        return (
            f"{self.checkpoint_root_path}/{exp_id}-best-weights/best_model_state_dict"
        )

    def __best_model_state_dict_filename_from_root(
        self, exp_id: str, fold: int, epoch: int, val_loss: float, val_jaccard: float
    ) -> str:
        return (
            f"{self.__best_model_state_dict_directory_from_root(exp_id=exp_id)}/"
            f"model_state_dict_{fold}_{epoch}_{val_loss:.4f}_{val_jaccard:.4f}.pkl"
        )

    def __checkpoint_filepath_from_root_to_val_jaccard(
        self, checkpoint_filepath_from_root: str
    ) -> float:
        score = float(
            checkpoint_filepath_from_root.split("/")[-1].split("_")[2][:-4]
        )  # -4 to remove .pkl
        return score

    def filepath_with_dataset_root(self, filepath_from_dataset_root: str) -> str:
        return f"{self.dataset_root_path}/{filepath_from_dataset_root}"

    def load_train_df(self) -> DataFrame:
        filepath_from_root = self.__train_df_filepath_from_root()
        df: DataFrame = self.load(
            filepath_from_root=filepath_from_root,
            mode="dfcsv",
            load_from_gcs=True,
            rm_local_after_load=False,
        )
        return df

    def load_cleaned_train_df(self) -> DataFrame:
        filepath_from_root = self.__cleaned_train_df_filepath_from_root()
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

    def load_booster_train_dfs(
        self, booster_train_filepaths: List[str]
    ) -> Dict[str, DataFrame]:
        booster_train_dfs = {}
        for booster_train_filepath in booster_train_filepaths:
            booster_train_filename = self.__filepath_to_filename_wo_extension(
                booster_train_filepath
            )
            booster_train_dfs[booster_train_filename] = self.load(
                filepath_from_root=booster_train_filepath,
                mode="dfcsv",
                load_from_gcs=True,
                rm_local_after_load=False,
            )
        return booster_train_dfs

    def preprocessed_df_exists(
        self,
        dataset_name: str,
        class_name: str,
        tokenizer_name: str,
        max_length: int,
        pad_on_right: bool,
        stride: int,
        use_language_as_question: bool,
    ) -> bool:
        filepath_from_root = self.__preprocessed_df_filepath_from_root(
            dataset_name=dataset_name,
            class_name=class_name,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            pad_on_right=pad_on_right,
            stride=stride,
            use_language_as_question=use_language_as_question,
        )
        gcs_files = self.list_gcs_filepaths_from_root(prefix=filepath_from_root)
        if len(gcs_files) == 0:
            return False
        elif len(gcs_files) == 1:
            return True
        else:
            raise Exception("should not occur case.")

    def save_preprocessed_df(
        self,
        dataset_name: str,
        preprocessed_df: DataFrame,
        class_name: str,
        tokenizer_name: str,
        max_length: int,
        pad_on_right: bool,
        stride: int,
        use_language_as_question: bool,
    ) -> None:
        filepath_from_root = self.__preprocessed_df_filepath_from_root(
            dataset_name=dataset_name,
            class_name=class_name,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            pad_on_right=pad_on_right,
            stride=stride,
            use_language_as_question=use_language_as_question,
        )
        self.save(
            save_obj=preprocessed_df,
            filepath_from_root=filepath_from_root,
            mode="pkl",
            gcs_mode="cp",
            force_save=True,
        )

    def load_preprocessed_df(
        self,
        dataset_name: str,
        class_name: str,
        tokenizer_name: str,
        max_length: int,
        pad_on_right: bool,
        stride: int,
        use_language_as_question: bool,
    ) -> DataFrame:
        filepath_from_root = self.__preprocessed_df_filepath_from_root(
            dataset_name=dataset_name,
            class_name=class_name,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            pad_on_right=pad_on_right,
            stride=stride,
            use_language_as_question=use_language_as_question,
        )
        self.logger.info(f"loading {filepath_from_root} ...")
        df = self.load(
            filepath_from_root=filepath_from_root,
            mode="pkl",
            load_from_gcs=True,
            rm_local_after_load=False,
        )
        self.logger.info("done.")
        return df

    def download_best_model_state_dicts(self, exp_id: str) -> None:
        target_dir = self.__best_model_state_dict_directory_from_root(exp_id=exp_id)
        filepaths_from_root = self.list_gcs_filepaths_from_root(prefix=target_dir)
        for filepath_from_root in filepaths_from_root:
            self.download(filepath_from_root=filepath_from_root)

    def iter_kaggle_kernel_best_model_state_dict(
        self, exp_id: str
    ) -> Generator[Dict[str, Tensor], None, None]:
        model_state_dict_dir = self.__best_kaggle_kernel_model_state_dict_directory_from_root(
            exp_id=exp_id
        )
        print(f"model_state_dict_dir: {model_state_dict_dir}")
        model_state_dict_filenames = self.list_local_filepaths_from_root(
            prefix=model_state_dict_dir
        )
        for model_state_dict_filename in model_state_dict_filenames:
            yield self.load(
                filepath_from_root=model_state_dict_filename,
                mode="pkl",
                load_from_gcs=False,
                rm_local_after_load=False,
            )

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
        # TODO: must use list_local_filepaths_from_root, and add _ to _filepath_with_local_root
        filepaths_with_local_root = glob(
            self._filepath_with_local_root(
                f"{self.checkpoint_root_path}/{exp_id}/{fold}/{epoch}_*"
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
        filepaths = self.list_gcs_filepaths_from_root(
            f"{self.checkpoint_root_path}/{exp_id}/best_checkpoint/"
        )
        return filepaths

    def clean_exp_checkpoint(self, exp_id: str, delete_from_gcs: bool) -> None:
        prefix = f"{self.checkpoint_root_path}/{exp_id}/"
        self.logger.info(f"now cleaning files from {prefix} ...")
        filepaths = self.list_gcs_filepaths_from_root(prefix)
        for filepath in filepaths:
            self.delete(
                filepath_from_root=filepath,
                delete_from_local=True,
                delete_from_gcs=delete_from_gcs,
            )
        rest_local_filepaths = self.list_local_filepaths_from_root(prefix)
        for rest_local_filepath in rest_local_filepaths:
            self.delete(
                filepath_from_root=rest_local_filepath,
                delete_from_local=True,
                delete_from_gcs=False,
            )
        self.logger.info("done.")

    @class_dec_timer(unit="m")
    def extract_and_save_best_fold_epoch_model_state_dict(
        self, exp_id: str, fold: int
    ) -> None:
        # gcs_filepaths = self.list_gcs_files(f"data/checkpoint/{exp_id}/{fold}/")
        filepaths = glob(f"{self.checkpoint_root_path}/{exp_id}/{fold}/*")
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
