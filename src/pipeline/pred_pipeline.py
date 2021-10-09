from glob import glob
from typing import Any, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm.auto import tqdm

from src.data.repository import DataRepository
from src.dataset.factory import DatasetFactory
from src.loader.factory import DatasetFactory, LoaderFactory
from src.log import myLogger
from src.model.factory import ModelFactory
from src.model.repository import ModelRepository
from src.pipeline.pipeline import Pipeline
from src.sampler.factory import SamplerFactory
from src.timer import class_dec_timer


class PredPipeline(Pipeline):
    def __init__(
        self,
        exp_id: str,
        config: Dict[str, Any],
        device: str,
        debug: bool,
        logger: myLogger,
        pipeline_type: str = "pred",
    ) -> None:
        super().__init__(pipeline_type, exp_id, logger)
        self.device = device
        self.debug = debug

        self.data_repository = DataRepository(logger=logger)
        self.model_repository = ModelRepository(logger=logger)

        self.loader_factory = LoaderFactory(**config["loader"], logger=logger)
        self.dataset_factory = DatasetFactory(**config["dataset"], logger=logger)
        self.sampler_factory = SamplerFactory(**config["sampler"], logger=logger)
        self.model_factory = ModelFactory(**config["model"], logger=logger)

    @class_dec_timer(unit="m")
    def run(self) -> None:
        self._pred()

    @class_dec_timer(unit="m")
    def _pred(self) -> None:
        loader = self._build_loader()
        model = self.model_factory.create()

        y_pred_probas = []
        for model_weight in tqdm(self.model_repository.fold_best_weights(self.exp_id)):
            model.load_weights(weight_path)
            y_pred_proba = model.predict(test_dataset)
            y_pred_probas.append(y_pred_proba)

        sub_df = self.data_repository.load_sample_submission_df()
        sub_df["target"] = np.mean(y_pred_probas, axis=1)

        self.data_repository.save_sub_df(exp_id=self.exp_id, sub_df=sub_df)

    def _build_loader(
        self,
        df: DataFrame,
        sampler_type: str,
        dataset_type: str,
        batch_size: int,
        drop_last: bool,
    ) -> Loader:
        dataset = self.dataset_factory.create(dataset_type)

    def _pred_loop(
        self,
    ):
        1
