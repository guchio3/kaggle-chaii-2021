import gc
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from src.config import ConfigLoader
from src.dataset.factory import DatasetFactory
from src.log import myLogger
from src.model.factory import ModelFactory
from src.model.model import Model
from src.pipeline.pipeline import Pipeline
from src.postprocessor.factory import PostprocessorFactory
from src.prediction.prediction_result import PredictionResult
from src.prediction.prediction_result_ensembler import (
    PredictionResultEnsembler, calc_id_to_context_len,
    ensemble_prediction_result, ensemble_prediction_results)
from src.preprocessor.factory import PreprocessorFactory
from src.repository.data_repository import DataRepository
from src.sampler.factory import SamplerFactory
from src.timer import class_dec_timer


class SubmissionPipeline(Pipeline):
    def __init__(
        self,
        exp_id: str,
        config: Dict[str, Any],
        device: str,
        data_origin_root_path: str,
        data_dataset_root_path: str,
        data_checkpoint_root_path: str,
        config_local_root_path: str,
        debug: bool,
        logger: myLogger,
    ) -> None:
        super().__init__("submission", exp_id, logger)
        self.config = config
        self.device = device
        self.debug = debug

        self.data_repository = DataRepository(
            origin_root_path=data_origin_root_path,
            dataset_root_path=data_dataset_root_path,
            checkpoint_root_path=data_checkpoint_root_path,
            logger=logger,
        )
        self.config_loader = ConfigLoader(local_root_path=config_local_root_path)

        self.train_exp_ids = config["train_exp_ids"]
        self.tst_batch_size = config["tst_batch_size"]
        self.ensemble_mode = config["ensemble_mode"]
        self.ensemble_weights = config["ensemble_weights"]

        self.postprocessor_factory = PostprocessorFactory(
            **config["postprocessor"], logger=logger
        )

        self.local_to_kaggle_kernel = {
            "data/dataset/deepset/xlm-roberta-large-squad2/": "../input/deepset/xlm-roberta-large-squad2/",
            "data/dataset/muril-large-cased/": "../input/muril-large-cased/",
        }

    def run(self) -> None:
        self._create_submission()

    @class_dec_timer(unit="m")
    def _create_submission(self) -> None:
        tst_df = self.data_repository.load_test_df()
        id_to_context_len = calc_id_to_context_len(df=tst_df)
        prediction_result_ensembler = PredictionResultEnsembler(
            id_to_context_len=id_to_context_len,
            ensemble_mode=self.ensemble_mode,
            logger=self.logger,
        )

        for train_exp_id in self.train_exp_ids:
            exp_train_config = self.config_loader.load(
                pipeline_type="train_pred", exp_id=train_exp_id, default_exp_id="e000"
            )

            # preprocessor
            exp_train_config["preprocessor"][
                "tokenizer_type"
            ] = self.local_to_kaggle_kernel[
                exp_train_config["preprocessor"]["tokenizer_type"]
            ]
            preprocessor_factory = PreprocessorFactory(
                **exp_train_config["preprocessor"], debug=self.debug, logger=self.logger
            )
            preprocessor = preprocessor_factory.create(
                data_repository=self.data_repository,
            )
            preprocessed_tst_df = preprocessor(
                df=tst_df, dataset_name="test", enforce_preprocess=False, is_test=True,
            )
            # loader
            dataset_factory = DatasetFactory(
                **exp_train_config["dataset"], logger=self.logger
            )
            sampler_factory = SamplerFactory(
                **exp_train_config["sampler"], logger=self.logger
            )
            tst_loader = self._build_loader(
                df=preprocessed_tst_df,
                dataset_factory=dataset_factory,
                sampler_factory=sampler_factory,
                sampler_type="sequential",
                batch_size=self.tst_batch_size,
                drop_last=False,
                debug=self.debug,
            )

            # model
            exp_train_config["model"][
                "pretrained_model_name_or_path"
            ] = self.local_to_kaggle_kernel[
                exp_train_config["model"]["pretrained_model_name_or_path"]
            ]
            model_factory = ModelFactory(
                **exp_train_config["model"], logger=self.logger
            )

            for (
                best_model_state_dict
            ) in self.data_repository.iter_kaggle_kernel_best_model_state_dict(
                exp_id=train_exp_id
            ):
                model = model_factory.create(order_settings=exp_train_config["model"])
                model.load_state_dict(best_model_state_dict)
                del best_model_state_dict
                gc.collect()
                prediction_result = self._predict(
                    device=self.device,
                    ensemble_weight=self.ensemble_weights[train_exp_id],
                    model=model,
                    loader=tst_loader,
                )
                del model
                gc.collect()
                ensemble_prediction_result(
                    prediction_result_ensembler=prediction_result_ensembler,
                    prediction_result=prediction_result,
                )
                del prediction_result
                gc.collect()
                # break
            gc.collect()

        ensembled_prediction_result = prediction_result_ensembler.to_prediction_result()
        del prediction_result_ensembler
        gc.collect()
        ensembled_prediction_result.sort_values_based_on_ids()
        ensembled_prediction_result.convert_elems_to_larger_level_as_possible()

        postprocessor = self.postprocessor_factory.create()
        contexts = (
            tst_df.set_index("id")
            .loc[ensembled_prediction_result.ids]["context"]
            .reset_index(drop=True)
            .tolist()
        )
        answer_texts = [""] * len(contexts)
        pospro_ids, _, pospro_answer_preds = postprocessor(
            ids=ensembled_prediction_result.ids,
            contexts=contexts,
            answer_texts=answer_texts,
            offset_mappings=ensembled_prediction_result.offset_mappings,
            start_logits=ensembled_prediction_result.start_logits,
            end_logits=ensembled_prediction_result.end_logits,
            segmentation_logits=ensembled_prediction_result.segmentation_logits,
        )
        sub_df = pd.DataFrame()
        sub_df["id"] = pospro_ids
        sub_df["PredictionString"] = pospro_answer_preds
        sub_df.to_csv("submission.csv", index=False)

    @class_dec_timer(unit="m")
    def _predict(
        self, device: str, ensemble_weight: float, model: Model, loader: DataLoader,
    ) -> PredictionResult:
        model.to(device)
        model.eval()

        prediction_result = PredictionResult(ensemble_weight=ensemble_weight)
        with torch.no_grad():
            for _, batch in enumerate(tqdm(loader)):
                ids = batch["id"]
                offset_mappings = batch["offset_mapping"]
                input_ids = batch["input_ids"].to(device)
                attention_masks = batch["attention_mask"].to(device)

                start_logits, end_logits, segmentation_logits = model(
                    input_ids=input_ids, attention_masks=attention_masks,
                )
                if start_logits.dim() == 1:
                    self.logger.info(
                        "fix the shape of logits because it contains just one elem."
                    )
                    start_logits = start_logits.reshape(1, -1)
                    end_logits = end_logits.reshape(1, -1)
                    segmentation_logits = segmentation_logits.reshape(1, -1)

                start_logits = start_logits.to("cpu")
                end_logits = end_logits.to("cpu")
                segmentation_logits = segmentation_logits.to("cpu")

                prediction_result.extend_by_value_list(key="ids", value_list=ids)
                prediction_result.extend_by_tensor(
                    key="offset_mappings", val_info=offset_mappings
                )
                prediction_result.extend_by_tensor(
                    key="start_logits", val_info=start_logits
                )
                prediction_result.extend_by_tensor(
                    key="end_logits", val_info=end_logits
                )
                prediction_result.extend_by_tensor(
                    key="segmentation_logits", val_info=segmentation_logits
                )

        model.to("cpu")

        return prediction_result

    def _build_loader(
        self,
        df: DataFrame,
        dataset_factory: DatasetFactory,
        sampler_factory: SamplerFactory,
        sampler_type: str,
        batch_size: int,
        drop_last: bool,
        debug: bool,
    ) -> DataLoader:
        if debug:
            df = df.iloc[: batch_size * 3]
        dataset = dataset_factory.create(df=df)
        sampler = sampler_factory.create(
            dataset=dataset, order_settings={"sampler_type": sampler_type}
        )
        _cpu_count = os.cpu_count()
        if self.debug or _cpu_count is None:
            num_workers = 1
        else:
            num_workers = _cpu_count
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            worker_init_fn=lambda _: np.random.seed(),
            drop_last=drop_last,
            pin_memory=True,
        )
        return loader


if __name__ == "__main__":
    logger = myLogger(
        log_filename="./logs/sub_log.log",
        exp_id="",
        wdb_prj_id="",
        exp_config={},
        use_wdb=False,
    )
    config = {
        "train_exp_ids": ["e016"],
        "tst_batch_size": 16,
        "ensemble_weights": {"e016": 1.0},
        "postprocessor": {
            "postprocessor_type": "baseline_kernel",
            "n_best_size": 20,
            "max_answer_length": 30,
        },
    }
    submission_pipeline = SubmissionPipeline(
        exp_id="",
        config=config,
        device="cuda",
        data_origin_root_path="data/origin",
        data_dataset_root_path="data/dataset",
        data_checkpoint_root_path="data/checkpoint",
        config_local_root_path="configs",
        debug=False,
        logger=logger,
    )
