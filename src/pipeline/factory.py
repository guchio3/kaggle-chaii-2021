from __future__ import annotations

from typing import Any, Dict

import yaml

from src.factory import Factory
from src.log import myLogger
from src.pipeline.pipeline import Pipeline
from src.pipeline.train_pred_pipeline import TrainPredPipeline


class PipelineFactory(Factory[Pipeline]):
    def _create(
        self,
        pipeline_type: str,
        mode: str,
        exp_id: str,
        device: str,
        enforce_preprocess: bool,
        debug: bool,
    ) -> Pipeline:
        config = self._load_config_from_yaml(pipeline_type, exp_id)
        default_config = self._load_config_from_yaml(pipeline_type, "e000")
        self._fill_config_by_default_config(config, default_config)

        use_wdb = self._use_wdb(pipeline_type=pipeline_type, mode=mode, debug=debug)
        logger = myLogger(
            log_filename=f"./logs/{pipeline_type}/{exp_id}.log",
            exp_id=exp_id,
            wdb_prj_id="kaggle-chaii-2021",
            exp_config=config,
            use_wdb=use_wdb,
        )

        if pipeline_type == "train_pred":
            pipeline = TrainPredPipeline(
                exp_id=exp_id,
                mode=mode,
                config=config,
                device=device,
                enforce_preprocess=enforce_preprocess,
                debug=debug,
                logger=logger,
            )
        else:
            raise NotImplementedError(f"pipeline {pipeline_type} is not supported yet.")

        return pipeline

    def _use_wdb(self, pipeline_type: str, mode: str, debug: bool) -> bool:
        if debug:
            return False
        if pipeline_type == "train_pred":
            if mode == "train":
                return True
            elif mode == "pred":
                return False

        raise Exception(
            f"invalid setting, pipeline_type: {pipeline_type} / mode: {mode} / debug: {debug}."
        )

    def _load_config_from_yaml(
        self, pipeline_type: str, exp_id: str
    ) -> Dict[str, Dict[str, Any]]:
        yaml_filename = f"./configs/{pipeline_type}/{exp_id}.yml"
        with open(yaml_filename, "r") as fin:
            config: Dict[str, Any] = yaml.load(fin, Loader=yaml.FullLoader)

        return config

    def _fill_config_by_default_config(
        self,
        config_dict: Dict[str, Dict[str, Any]],
        default_config_dict: Dict[str, Dict[str, Any]],
    ) -> None:
        for (d_key, d_value) in default_config_dict.items():
            if d_key not in config_dict:
                config_dict[d_key] = d_value
            elif isinstance(d_value, dict):
                self._fill_config_by_default_config(config_dict[d_key], d_value)
