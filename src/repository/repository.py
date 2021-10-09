import os
import pickle
from dataclasses import dataclass
from typing import Any, List

import pandas as pd
from google.cloud import storage
from pandas import DataFrame

from src.log import myLogger


@dataclass(frozen=True)
class Repository:
    logger: myLogger
    bucket_name: str

    def save(
        self,
        save_obj: Any,
        filepath: str,
        mode: str,
        gcs_mode: str = "pass",
        force_save: bool = False,
    ) -> None:
        if os.path.exists(filepath) and not force_save:
            self.logger.info(f"{filepath} already exists.")
            self.logger.info("please use force_save if you wanna save it.")
            return

        file_dir = "/".join(filepath.split("/")[:-1])
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        if mode == "dfcsv":
            self.__save_dfscv(df=save_obj, filepath=filepath)
        elif mode == "pkl":
            self.__save_pickle(save_obj=save_obj, filepath=filepath)
        else:
            raise NotImplementedError(f"mode {mode} is not supported.")

        if gcs_mode in ["cp", "mv"]:
            self.__upload_to_gcs(src_filepath=filepath, dst_filepath=filepath)
        else:
            raise NotImplementedError(f"gcs_mode {gcs_mode} is not supported.")
        if gcs_mode == "mv":
            os.remove(filepath)

    def __upload_to_gcs(self, src_filepath: str, dst_filepath: str) -> None:
        self.logger.info(
            f"upload {src_filepath} to gs://{self.bucket_name}/{dst_filepath}"
        )
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(dst_filepath)
        blob.upload_from_filename(src_filepath)
        self.logger.info("done.")

    def __save_dfscv(self, df: DataFrame, filepath: str) -> None:
        df.to_csv(filepath, index=False)

    def __save_pickle(self, save_obj: Any, filepath: str) -> None:
        with open(filepath, "rb") as fout:
            pickle.dump(save_obj, fout)

    def load(self, filepath: str, mode: str, load_from_gcs: bool = False) -> Any:
        if not os.path.exists(filepath) and load_from_gcs:
            self.__download_from_gcs(src_filepath=filepath, dst_filepath=filepath)

        if mode == "dfcsv":
            res = self.__load_dfcsv(filepath)
        elif mode == "pkl":
            res = self.__load_pickle(filepath)
        else:
            raise NotImplementedError(f"mode {mode} is not supported.")
        return res

    def __download_from_gcs(self, src_filepath: str, dst_filepath: str) -> None:
        self.logger.info(
            f"download {src_filepath} from gs://{self.bucket_name}/{dst_filepath}"
        )
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(src_filepath)
        blob.download_to_filename(dst_filepath)
        self.logger.info("done.")

    def __load_dfcsv(self, filepath: str) -> DataFrame:
        df: DataFrame = pd.read_csv(filepath)
        return df

    def __load_pickle(self, filepath: str) -> Any:
        with open(filepath, "rb") as fin:
            res = pickle.load(fin)
        return res

    def list_gcs_files(self, prefix: str) -> List[str]:
        storage_client = storage.Client()

        # Note: Client.list_blobs requires at least package version 1.17.0.
        blobs = storage_client.list_blobs(
            self.bucket_name, prefix=prefix, delimiter=None
        )
        res = [f"gs://{self.bucket_name}/{blob.name}" for blob in blobs]

        return res

    def delete_gcs_file(self, prefix: str) -> None:
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(prefix)
        blob.delete()
