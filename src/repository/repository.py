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
        gcs_mode: str,
        force_save: bool
    ):
        1

    def __upload_to_gcs(self, src_filepath: str, dst_filepath: str) -> None:
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(dst_filepath)
        blob.upload_from_filename(src_filepath)

    def __save_dfscv(self, df: DataFrame, filepath: str) -> None:
        df.to_csv(filepath, index=False)

    def __save_pickle(self, save_obj: Any, filepath: str) -> None:
        with open(filepath, "rb") as fout:
            pickle.dump(save_obj, fout)

    def load(
        self,
    ):
        1

    def __download_from_gcs(self, src_filepath: str, dst_filepath: str) -> None:
        storage_client = storage.Client()
        bucket = storage_client.bucket(self.bucket_name)
        blob = bucket.blob(src_filepath)
        blob.download_to_filename(dst_filepath)

    def __load_dfcsv(self, filepath: str) -> DataFrame:
        df: DataFrame = pd.read_csv(filepath)
        return df

    def __load_pickle(self, filepath: str) -> Any:
        with open(filepath, "rb") as fin:
            res = pickle.load(fin)
        return res

    def list_gs_files(self, prefix: str) -> List[str]:    
        storage_client = storage.Client()    

        # Note: Client.list_blobs requires at least package version 1.17.0.    
        blobs = storage_client.list_blobs(    
            self.bucket_name, prefix=prefix, delimiter=None    
        )    
        res = [f"gs://{self.bucket_name}/{blob.name}" for blob in blobs]    

        return res   
