from dataclasses import dataclass

from numpy import ndarray
from pandas import DataFrame

from src.log import myLogger
from src.repository.repository import Repository


@dataclass(frozen=True)
class DataRepository(Repository):
    logger: myLogger
    origin_data_dir: str = "./data/origin"
    train_filename: str = "train.csv"
    test_filename: str = "test.csv"
    sample_submission_filename: str = "sample_submission.csv"
    bucket_name: str = "kaggle-chaii-2021"

    def load_train_df(self) -> DataFrame:
        filepath = f"{self.origin_data_dir}/{self.train_filename}"
        df: DataFrame = self.load(filepath=filepath, mode="dfcsv", load_from_gcs=True)
        return df

    def load_test_df(self) -> DataFrame:
        filepath = f"{self.origin_data_dir}/{self.test_filename}"
        df: DataFrame = self.load(filepath=filepath, mode="dfcsv", load_from_gcs=True)
        return df

    def load_sample_submission_df(self) -> DataFrame:
        filepath = f"{self.origin_data_dir}/{self.sample_submission_filename}"
        df: DataFrame = self.load(filepath=filepath, mode="dfcsv")
        return df

    def save_sub_df(self, sub_df: DataFrame, exp_id: str) -> None:
        filepath = f"./data/submissions/sub_{exp_id}.csv"
        self.save(
            save_obj=sub_df,
            filepath=filepath,
            mode="dfcsv",
            gcs_mode="pass",
            force_save=True,
        )

    def load_sub_df(self, exp_id: str) -> DataFrame:
        filepath = f"./data/submissions/sub_{exp_id}.csv"
        df = self.load(filepath=filepath, mode="dfcsv", load_from_gcs=False)
        return df

    def save_preprocessed_df(self, preprocessed_df: DataFrame, ver: str) -> None:
        filepath = f"./data/preprocessed/{ver}.csv"
        self.save(
            save_obj=preprocessed_df,
            filepath=filepath,
            mode="dfcsv",
            gcs_mode="cp",
        )

    def load_preprocessed_df(self, ver: str) -> DataFrame:
        filepath = f"./data/preprocessed/{ver}.csv"
        df = self.load(filepath=filepath, mode="dfcsv", load_from_gcs=True)
        return df

    def save_fold_idxes(self, exp_id: str, fold: int, fold_idxes: ndarray) -> None:
        filepath = f"./data/fold/{exp_id}/{fold}.npy"
        self.save(
            save_obj=fold_idxes,
            filepath=filepath,
            mode="np",
            gcs_mode="cp",
            force_save=False,
        )

    def load_fold_idxes(self, exp_id: str, fold: int) -> ndarray:
        filepath = f"./data/fold/{exp_id}/{fold}.npy"
        fold_idxes = self.load(filepath=filepath, mode="np", load_from_gcs=True)
        return fold_idxes
