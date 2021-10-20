import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, List

import numpy as np
import pandas as pd
import torch
from google.cloud import storage
from numpy import ndarray
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
        if not force_save and self.list_gcs_files(prefix=filepath):
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
        elif mode == "json":
            self.__save_json(save_obj=save_obj, filepath=filepath)
        elif mode == "np":
            self.__save_numpy(save_obj=save_obj, filepath=filepath)
        elif mode == "pt":
            self.__save_torch(save_obj=save_obj, filepath=filepath)
        else:
            raise NotImplementedError(f"mode {mode} is not supported.")

        if gcs_mode == "pass":
            pass
        elif gcs_mode in ["cp", "mv"]:
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
        with open(filepath, "wb") as fout:
            pickle.dump(save_obj, fout)

    def __save_json(self, save_obj: Any, filepath: str) -> None:
        with open(filepath, "w") as fout:
            json.dump(save_obj, fout)

    def __save_numpy(self, save_obj: ndarray, filepath: str) -> None:
        np.save(filepath, save_obj)

    def __save_torch(self, save_obj: Any, filepath: str) -> None:
        torch.save(save_obj, filepath)

    def load(
        self,
        filepath: str,
        mode: str,
        load_from_gcs: bool = False,
        rm_local_after_load: bool = False,
    ) -> Any:
        if not os.path.exists(filepath):
            if load_from_gcs:
                self.__download_from_gcs(src_filepath=filepath, dst_filepath=filepath)
            else:
                raise Exception(f"{filepath} does not exist int local.")

        if mode == "dfcsv":
            res = self.__load_dfcsv(filepath)
        elif mode == "pkl":
            res = self.__load_pickle(filepath)
        elif mode == "json":
            res = self.__load_json(filepath)
        elif mode == "np":
            res = self.__load_numpy(filepath)
        elif mode == "pt":
            res = self.__load_torch(filepath)
        else:
            raise NotImplementedError(f"mode {mode} is not supported.")

        if rm_local_after_load:
            os.remove(filepath)
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

    def __load_json(self, filepath: str) -> Any:
        with open(filepath, "r") as fin:
            res = json.load(fin)
        return res

    def __load_numpy(self, filepath: str) -> ndarray:
        res: ndarray = np.load(filepath)
        return res

    def __load_torch(self, filepath: str) -> ndarray:
        res = torch.load(filepath)
        return res

    def list_gcs_files(self, prefix: str) -> List[str]:
        storage_client = storage.Client()

        # Note: Client.list_blobs requires at least package version 1.17.0.
        blobs = storage_client.list_blobs(
            self.bucket_name, prefix=prefix, delimiter=None
        )
        res = [f"gs://{self.bucket_name}/{blob.name}" for blob in blobs]

        return res

    def delete(self, filepath: str, delete_from_local: bool, delete_from_gcs: bool) -> None:
        if delete_from_local:
            if os.path.exists(filepath):
                os.remove(filepath)
            else:
                self.logger.warn(
                    f"ignore deleting local {filepath} because it does not exist."
                )

        if delete_from_gcs:
            gcs_file = self.list_gcs_files(prefix=filepath)
            if len(gcs_file) == 1:
                self.delete_gcs_file(prefix=filepath)
            else:
                self.logger.warn(
                    f"ignore deleting gcs {filepath} because it does not exist."
                )

    def delete_gcs_file(self, prefix: str) -> None:
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(prefix)
        blob.delete()

    def copy_gcs_file(self, src_filepath: str, dst_filepath: str) -> None:
        storage_client = storage.Client()
        src_bucket = storage_client.bucket(self.bucket_name)
        src_blob = src_bucket.blob(src_filepath)
        dst_bucket = storage_client.bucket(self.bucket_name)
        _ = src_bucket.copy_blob(src_blob, dst_bucket, dst_filepath)

    def move_gcs_file(self, src_filepath: str, dst_filepath: str) -> None:
        self.copy_gcs_file(src_filepath=src_filepath, dst_filepath=dst_filepath)
        self.delete_gcs_file(prefix=src_filepath)
