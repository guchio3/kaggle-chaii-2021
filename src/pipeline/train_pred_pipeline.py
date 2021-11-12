import os
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
# from torch.nn import DataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from src.checkpoint.checkpoint import Checkpoint
from src.dataset.factory import DatasetFactory
# from src.error_handling import class_error_line_notification
from src.fobj.factory import FobjFactory
from src.log import myLogger
from src.model.factory import ModelFactory
from src.model.model import Model
from src.optimizer.factory import OptimizerFactory
from src.pipeline.pipeline import Pipeline
from src.postprocessor.factory import PostprocessorFactory
from src.preprocessor.factory import PreprocessorFactory
from src.repository.data_repository import DataRepository
from src.sampler.factory import SamplerFactory
from src.scheduler.factory import SchedulerFactory
from src.splitter.factory import SplitterFactory
from src.timer import class_dec_timer


class TrainPredPipeline(Pipeline):
    def __init__(
        self,
        exp_id: str,
        config: Dict[str, Any],
        device: str,
        enforce_preprocess: bool,
        debug: bool,
        logger: myLogger,
    ) -> None:
        super().__init__("train_pred", exp_id, logger)
        self.config = config
        self.device = device
        self.enforce_preprocess = enforce_preprocess
        self.debug = debug

        self.data_repository = DataRepository(logger=logger)

        self.all_data_train = config["all_data_train"]
        self.cleaned_train = config["cleaned_train"]
        self.negative_sampling_num = config["negative_sampling_num"]
        self.only_answer_text_training = config["only_answer_text_training"]
        self.only_answer_text_validation = config["only_answer_text_validation"]
        self.max_answer_text_count = config["max_answer_text_count"]
        self.num_epochs = config["num_epochs"]
        self.train_folds = config["train_folds"]
        self.accum_mod = config["accum_mod"]
        self.trn_batch_size = config["trn_batch_size"]
        self.val_batch_size = config["val_batch_size"]
        self.tst_batch_size = config["tst_batch_size"]
        self.booster_trn_data = config["booster_trn_data"]
        self.schedule_per_batch = config["schedule_per_batch"]

        self.preprocessor_factory = PreprocessorFactory(
            **config["preprocessor"], debug=debug, logger=logger
        )
        self.postprocessor_factory = PostprocessorFactory(
            **config["postprocessor"], logger=logger
        )
        self.splitter_factory = SplitterFactory(**config["splitter"], logger=logger)
        self.dataset_factory = DatasetFactory(**config["dataset"], logger=logger)
        self.sampler_factory = SamplerFactory(**config["sampler"], logger=logger)
        self.model_factory = ModelFactory(**config["model"], logger=logger)
        self.optimizer_factory = OptimizerFactory(
            **config["optimizer"], logger=self.logger
        )
        self.fobj_factory = FobjFactory(**config["fobj"], logger=logger)
        self.scheduler_factory = SchedulerFactory(
            **config["scheduler"], logger=self.logger
        )

    def run(self) -> None:
        self._train()

    # @class_error_line_notification(add_traceback=True, return_value=None)
    @class_dec_timer(unit="m")
    def _train(self) -> None:
        # clean best model weights
        self.data_repository.clean_exp_checkpoint(
            exp_id=self.exp_id, delete_from_gcs=True
        )

        if self.cleaned_train:
            trn_df = self.data_repository.load_cleaned_train_df()
        else:
            trn_df = self.data_repository.load_train_df()
        trn_df["top20_context"] = trn_df["context"].apply(lambda context: context[:20])
        booster_train_dfs = self.data_repository.load_booster_train_dfs(
            self.booster_trn_data
        )
        preprocessor = self.preprocessor_factory.create(
            data_repository=self.data_repository
        )
        preprocessed_trn_df = preprocessor(
            df=trn_df,
            dataset_name="cleaned_train" if self.cleaned_train else "train",
            enforce_preprocess=self.enforce_preprocess,
            is_test=False,
        )
        preprocessed_booster_train_dfs = []
        for booster_dataset_name, booster_train_df in booster_train_dfs.items():
            preprocessed_booster_train_dfs.append(
                preprocessor(
                    df=booster_train_df,
                    dataset_name=booster_dataset_name,
                    enforce_preprocess=self.enforce_preprocess,
                    is_test=False,
                )
            )
        preprocessed_booster_train_df = pd.concat(
            preprocessed_booster_train_dfs, axis=0
        ).reset_index(drop=True)

        splitter = self.splitter_factory.create()
        folds = splitter.split(
            trn_df["id"], trn_df["language"], groups=trn_df["top20_context"]
        )

        best_val_jaccards = []
        for fold, (trn_idx, val_idx) in enumerate(folds):
            if fold not in self.train_folds:
                self.logger.info(f"skip fold {fold} because it's not in train_folds.")
                continue
            # fold data
            if self.all_data_train:
                fold_trn_df = preprocessed_trn_df.copy()
            else:
                trn_ids = trn_df.iloc[trn_idx]["id"].tolist()
                fold_trn_df = preprocessed_trn_df.query(f"id in {trn_ids}")
            fold_trn_df = pd.concat(
                [fold_trn_df, preprocessed_booster_train_df], axis=0
            ).reset_index(drop=True)
            if self.only_answer_text_training:
                fold_trn_df = fold_trn_df.query("is_contain_answer_text == 1")
            fold_trn_df = fold_trn_df.query(
                f"answer_text_count <= {self.max_answer_text_count} or part_answer_text_count == 0"
            )
            if self.negative_sampling_num > 0:
                self.logger.info("negative down sampling...")
                sampled_reses = []
                for _, grp_df in tqdm(fold_trn_df.groupby("id")):
                    sampled_reses.append(self._negative_down_sampling(grp_df=grp_df))
                fold_trn_df = pd.concat(sampled_reses, axis=0,).reset_index(drop=True)
            trn_loader = self._build_loader(
                df=fold_trn_df,
                sampler_type=self.config["sampler"]["trn_sampler_type"],
                batch_size=self.trn_batch_size,
                drop_last=True,
                debug=self.debug,
            )
            if not self.all_data_train:
                val_ids = trn_df.iloc[val_idx]["id"].tolist()
                fold_val_df = preprocessed_trn_df.query(f"id in {val_ids}")
                if self.only_answer_text_validation:
                    fold_val_df = fold_val_df.query("is_contain_answer_text == 1")
                val_loader = self._build_loader(
                    df=fold_val_df,
                    sampler_type=self.config["sampler"]["val_sampler_type"],
                    batch_size=self.val_batch_size,
                    drop_last=False,
                    debug=self.debug,
                )

            # fold model
            model, optimizer, scheduler = self._build_model(loader_size=len(trn_loader))
            fobj = self.fobj_factory.create()

            val_jaccards = []
            for epoch in range(self.num_epochs):
                model.train_one_epoch(
                    device=self.device,
                    fold=fold,
                    epoch=epoch,
                    accum_mod=self.accum_mod,
                    loader=trn_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    fobj=fobj,
                    segmentation_fobj=None,
                    schedule_per_batch=self.schedule_per_batch,
                )
                checkpoint = Checkpoint(exp_id=self.exp_id, fold=fold, epoch=epoch)
                checkpoint.set_model(model=model)
                checkpoint.set_optimizer(optimizer=optimizer)
                checkpoint.set_scheduler(scheduler=scheduler)
                if not self.all_data_train:
                    postprocessor = self.postprocessor_factory.create()
                    model.valid(
                        device=self.device,
                        fold=fold,
                        epoch=epoch,
                        loader=val_loader,
                        fobj=fobj,
                        segmentation_fobj=None,
                        postprocessor=postprocessor,
                        checkpoint=checkpoint,
                    )
                else:
                    checkpoint.val_loss = 0.0
                    checkpoint.val_jaccard = epoch
                val_jaccards.append(checkpoint.val_jaccard)
                if not self.debug:
                    self.data_repository.save_checkpoint(
                        checkpoint=checkpoint, is_best=False
                    )
            best_val_jaccard = max(val_jaccards)
            best_val_jaccards.append(best_val_jaccard)
            fold_result_stat = f"best_val_jaccard for exp_id {self.exp_id} fold {fold} : {best_val_jaccard:.5f}"
            self.logger.info(fold_result_stat)
            self.logger.send_line_notification(fold_result_stat)

            if not self.debug:
                self.data_repository.extract_and_save_best_fold_epoch_model_state_dict(
                    exp_id=self.exp_id, fold=fold
                )
            if self.all_data_train:
                self.logger.info(
                    "break on the first epoch, because all_data_train mode."
                )
                break

        val_jaccard_mean = np.mean(best_val_jaccards)
        val_jaccard_std = np.std(best_val_jaccards)
        self.logger.wdb_sum(
            sum_dict={
                "val_jaccard_mean": val_jaccard_mean,
                "val_jaccard_std": val_jaccard_std,
            }
        )
        result_stats = (
            f"finish train for {self.exp_id} !!!\n"
            f"val_jaccard_mean: {val_jaccard_mean} / val_jaccard_std: {val_jaccard_std}"
        )
        self.logger.info(result_stats)
        self.logger.send_line_notification(message=result_stats)

    def _negative_down_sampling(self, grp_df: DataFrame) -> DataFrame:
        positive_samples_df = grp_df.query("is_contain_answer_text == 1")
        tmp_negative_samples = grp_df.query(
            "is_contain_answer_text == 0 and part_answer_text_count == 0"
        )
        negative_samples_df = tmp_negative_samples.sample(
            min(self.negative_sampling_num, len(tmp_negative_samples)), random_state=71
        )
        negative_down_sampled_df = pd.concat(
            [positive_samples_df, negative_samples_df], axis=0
        )
        return negative_down_sampled_df

    def _build_loader(
        self,
        df: DataFrame,
        sampler_type: str,
        batch_size: int,
        drop_last: bool,
        debug: bool,
    ) -> DataLoader:
        if debug:
            df = df.iloc[: batch_size * 3]
        dataset = self.dataset_factory.create(df=df, is_test=False)
        sampler = self.sampler_factory.create(
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
            # num_workers=1,
            worker_init_fn=lambda _: np.random.seed(),
            drop_last=drop_last,
            pin_memory=True,
        )
        return loader

    def _build_model(self, loader_size: int) -> Tuple[Model, Optimizer, _LRScheduler]:
        model = self.model_factory.create()
        optimizer = self.optimizer_factory.create(model=model)
        scheduler = self.scheduler_factory.create(
            optimizer=optimizer,
            num_epochs=self.num_epochs,
            loader_size=loader_size,
            accum_mod=self.accum_mod,
            schedule_per_batch=self.schedule_per_batch,
        )
        return model, optimizer, scheduler
