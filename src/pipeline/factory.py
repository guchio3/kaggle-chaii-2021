from __future__ import annotations

from src.config import ConfigLoader
from src.factory import Factory
from src.log import myLogger
from src.pipeline.pipeline import Pipeline
from src.pipeline.text_batch_choose_train_pipeline import \
    TextBatchChooseTrainPipeline
from src.pipeline.train_pred_pipeline import TrainPredPipeline


class PipelineFactory(Factory[Pipeline]):
    def _create(
        self,
        pipeline_type: str,
        exp_id: str,
        device: str,
        enforce_preprocess: bool,
        local_root_path: str,
        debug: bool,
    ) -> Pipeline:
        config_loader = ConfigLoader(local_root_path=local_root_path)
        config = config_loader.load(
            pipeline_type=pipeline_type, exp_id=exp_id, default_exp_id="e000"
        )

        use_wdb = self._use_wdb(pipeline_type=pipeline_type, debug=debug)
        logger = myLogger(
            log_filename=f"{local_root_path}/logs/{pipeline_type}/{exp_id}.log",
            exp_id=exp_id,
            wdb_prj_id="kaggle-chaii-2021",
            exp_config=config,
            use_wdb=use_wdb,
        )

        if pipeline_type == "train_pred":
            pipeline = TrainPredPipeline(
                exp_id=exp_id,
                config=config,
                device=device,
                enforce_preprocess=enforce_preprocess,
                debug=debug,
                logger=logger,
            )
        elif pipeline_type == "text_batch":
            pipeline = TextBatchChooseTrainPipeline(
                exp_id=exp_id,
                config=config,
                device=device,
                enforce_preprocess=enforce_preprocess,
                debug=debug,
                logger=logger,
            )
        else:
            raise NotImplementedError(f"pipeline {pipeline_type} is not supported yet.")

        return pipeline

    def _use_wdb(self, pipeline_type: str, debug: bool) -> bool:
        if debug:
            return False
        if pipeline_type == "train_pred":
            return True
        if pipeline_type == "text_batch":
            return True

        raise Exception(
            f"invalid setting, pipeline_type: {pipeline_type} / debug: {debug}."
        )
