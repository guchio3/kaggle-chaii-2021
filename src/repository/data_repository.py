from dataclasses import dataclass

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
            gcs_mode="cp",
            force_save=True,
        )

    def save_preprocessed_df(self, preprocessed_df: DataFrame, ver: str) -> None:
        filepath = f"./data/preprocessed/{ver}.csv"
        self.save(
            save_obj=preprocessed_df,
            filepath=filepath,
            mode="dfcsv",
            gcs_mode="cp",
        )
