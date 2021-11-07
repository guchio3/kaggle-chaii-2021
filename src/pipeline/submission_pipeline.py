import gc
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data.dataloader import DataLoader

from src.config import ConfigLoader
from src.dataset.factory import DatasetFactory
from src.log import myLogger
from src.model.chaii_textbatch_model import ChaiiTextBatchXLMRBModel1
from src.model.factory import ModelFactory
from src.pipeline.pipeline import Pipeline
from src.postprocessor.factory import PostprocessorFactory
from src.prediction.prediction_result_ensembler import (
    PredictionResultEnsembler, SimplePredictionResultEnsembler,
    calc_id_to_context_len, ensemble_prediction_result)
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
        enforce_split: bool,
        debug: bool,
        logger: myLogger,
    ) -> None:
        super().__init__("submission", exp_id, logger)
        self.config = config
        self.device = device
        self.enforce_split = enforce_split
        self.debug = debug

        self.data_repository = DataRepository(
            origin_root_path=data_origin_root_path,
            dataset_root_path=data_dataset_root_path,
            checkpoint_root_path=data_checkpoint_root_path,
            logger=logger,
        )
        self.config_loader = ConfigLoader(local_root_path=config_local_root_path)

        self.train_exp_ids = config["train_exp_ids"]
        self.text_batch_exp_ids = config["text_batch_exp_ids"]
        self.tst_batch_size = config["tst_batch_size"]
        self.ensembler_type = config["ensembler_type"]
        self.ensemble_mode = config["ensemble_mode"]
        self.ensemble_textbatch_max_length = config["ensemble_textbatch_max_length"]
        self.ensemble_textbatch_stride = config["ensemble_textbatch_stride"]
        self.ensemble_weights = config["ensemble_weights"]
        self.textbatch_ensemble_weights = config["textbatch_ensemble_weights"]
        self.text_batch_topn = config["text_batch_topn"]

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
        if self.enforce_split:
            tst_df["context"] = tst_df["context"].apply(lambda x: " ".join(x.split()))
            tst_df["question"] = tst_df["question"].apply(lambda x: " ".join(x.split()))
        id_to_context_len = calc_id_to_context_len(df=tst_df)
        if self.ensembler_type == "simple":
            prediction_result_ensembler = SimplePredictionResultEnsembler(
                logger=self.logger,
            )
        elif self.ensembler_type == "char_level":
            prediction_result_ensembler = PredictionResultEnsembler(
                id_to_context_len=id_to_context_len,
                ensemble_mode=self.ensemble_mode,
                logger=self.logger,
            )
        else:
            raise Exception("ensembler_type {self.ensembler_type} is invalid.")

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
            del preprocessor
            del preprocessor_factory
            gc.collect()
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
            if len(self.text_batch_exp_ids) > 0:
                ensembled_text_batch_logits = None
                for text_batch_exp_id in self.text_batch_exp_ids:
                    text_batch_config = self.config_loader.load(
                        pipeline_type="text_batch",
                        exp_id=text_batch_exp_id,
                        default_exp_id="e000",
                    )
                    text_batch_config["model"][
                        "pretrained_model_name_or_path"
                    ] = self.local_to_kaggle_kernel[
                        text_batch_config["model"]["pretrained_model_name_or_path"]
                    ]
                    model_factory = ModelFactory(
                        **text_batch_config["model"], logger=self.logger
                    )
                    for (
                        best_model_state_dict
                    ) in self.data_repository.iter_kaggle_kernel_best_model_state_dict(
                        exp_id=text_batch_exp_id
                    ):
                        model: ChaiiTextBatchXLMRBModel1 = model_factory.create(
                            order_settings=text_batch_config["model"]
                        )
                        model.load_state_dict(best_model_state_dict)
                        del best_model_state_dict
                        gc.collect()
                        text_batch_logits = model.textbatch_predict(
                            device=self.device,
                            ensemble_weight=self.textbatch_ensemble_weights[
                                text_batch_exp_id
                            ],
                            loader=tst_loader,
                        )
                        del model
                        gc.collect()
                        if ensembled_text_batch_logits is None:
                            ensembled_text_batch_logits = text_batch_logits
                        else:
                            ensembled_text_batch_logits += text_batch_logits
                        del text_batch_logits
                        gc.collect()
                    del model_factory
                    gc.collect()
                preprocessed_tst_df["text_batch_logits"] = ensembled_text_batch_logits
                del ensembled_text_batch_logits
                gc.collect()

                preprocessed_tst_df = (
                    preprocessed_tst_df.groupby("id")
                    .apply(
                        lambda grp_df: grp_df.sort_values(
                            "text_batch_logits", ascending=False
                        ).head(self.text_batch_topn)
                    )
                    .reset_index(drop=True)
                )
                del tst_loader
                gc.collect()
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
                prediction_result = model.predict(
                    device=self.device,
                    ensemble_weight=self.ensemble_weights[train_exp_id],
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
            del model_factory
            del preprocessed_tst_df
            del tst_loader
            torch.cuda.empty_cache()
            gc.collect()

        del id_to_context_len
        gc.collect()

        ensembled_prediction_result = prediction_result_ensembler.to_prediction_result()
        del prediction_result_ensembler
        gc.collect()
        ensembled_prediction_result.sort_values_based_on_ids()
        ensembled_prediction_result.convert_elems_to_larger_level_as_possible()
        if (
            self.ensemble_textbatch_max_length > 0
            and self.ensemble_textbatch_stride > 0
        ):
            if self.ensembler_type == "simple":
                raise Exception("to_textbatched cannot be used w/ simple ensembler.")
            self.logger.info("!!!!!!!!!!!!! apply to_textbatched !!!!!!!!!!!!!")
            ensembled_prediction_result.to_textbatched(
                max_length=self.ensemble_textbatch_max_length,
                stride=self.ensemble_textbatch_stride,
            )
        else:
            self.logger.info("!!!!!!!!!!!!! skip to_textbatched !!!!!!!!!!!!!")

        postprocessor = self.postprocessor_factory.create()
        contexts = (
            tst_df.set_index("id")
            .loc[ensembled_prediction_result.ids]["context"]
            .reset_index(drop=True)
            .tolist()
        )
        answer_texts = [""] * len(contexts)
        del tst_df
        gc.collect()
        pospro_ids, _, pospro_answer_preds = postprocessor(
            ids=ensembled_prediction_result.ids,
            contexts=contexts,
            answer_texts=answer_texts,
            offset_mappings=ensembled_prediction_result.offset_mappings,
            start_logits=ensembled_prediction_result.start_logits,
            end_logits=ensembled_prediction_result.end_logits,
        )
        del postprocessor
        del ensembled_prediction_result
        gc.collect()
        sub_df = pd.DataFrame()
        sub_df["id"] = pospro_ids
        sub_df["PredictionString"] = pospro_answer_preds
        sub_df.to_csv("submission.csv", index=False)

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
        dataset = dataset_factory.create(df=df, is_test=True)
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
