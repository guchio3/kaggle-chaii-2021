import os
from glob import glob
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from torch.nn import DataParallel, Module, _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.dataset.factory import DatasetFactory
from src.fobj.factory import FobjFactory
from src.loader.factory import DatasetFactory, LoaderFactory
from src.log import myLogger
from src.model.factory import ModelFactory
from src.model.repository import ModelRepository
from src.optimizer.factory import OptimizerFactory
from src.pipeline.pipeline import Pipeline
from src.preprocessor.factory import PreprocessorFactory
from src.repository.repository import DataRepository
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
        mode: str,
        debug: bool,
        logger: myLogger,
    ) -> None:
        super().__init__("train_pred", exp_id, logger)
        self.config = config
        self.device = device
        self.mode = mode
        self.debug = debug

        self.data_repository = DataRepository(logger=logger)
        self.checkpoint_repository = CheckpointRepository(logger=logger)

        self.num_epochs = config["num_epochs"]

        self.preprocessor_factory = PreprocessorFactory(
            **config["preprocessor"], logger=logger
        )
        self.splitter_factory = SplitterFactory(**config["splitter"], logger=logger)
        self.loader_factory = LoaderFactory(**config["loader"], logger=logger)
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

    @class_dec_timer(unit="m")
    def run(self) -> None:
        if self.mode == "train":
            self._train()
        elif self.mode == "pred":
            self._pred()
        else:
            raise NotImplementedError(f"mode {self.mode} is not supported.")

    @class_dec_timer(unit="m")
    def _train(self) -> None:
        trn_df = self.data_repository.load_train_df()
        preprocessor = self.preprocessor_factory.create()
        trn_df = preprocessor(trn_df)

        splitter = self.splitter_factory.create()
        fold = splitter.split(
            trn_df["id"], trn_df["language"], groups=None
        )

        for fold_num, (trn_idx, val_idx) in enumerate(fold):
            # fold data
            fold_trn_df = trn_df.iloc[trn_idx]
            trn_loader = self._build_loader(
                df=fold_trn_df,
                sampler_type=self.config["loader"]["trn_sampler_type"],
                batch_size=self.config["loader"]["trn_batch_size"],
                drop_last=True,
                debug=self.debug,
            )
            fold_val_df = trn_df.iloc[val_idx]
            val_loader = self._build_loader(
                df=fold_val_df,
                sampler_type=self.config["loader"]["val_sampler_type"],
                batch_size=self.config["loader"]["val_batch_size"],
                drop_last=True,
                debug=self.debug,
            )

            # fold model
            model, optimizer, scheduler = self._build_model()
            fobj = self.fobj_factory.create()

            for epoch in range(self.num_epochs):
                self._train_one_epoch(
                    device=self.device,
                    epoch=epoch,
                    loader=trn_loader,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    fobj=fobj,
                )
                self._valid(
                    fold_num=fold_num, model=model, loader=val_loader, fobj=fobj
                )
                self.checkpoint_repository.save(
                    fold_num=fold_num,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    debug=self.debug,
                )

    @class_dec_timer(unit="m")
    def _train_one_epoch(
        self,
        device: str,
        epoch: int,
        loader: DataLoader,
        model: Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        fobj: _Loss,
    ) -> None:
        # init for train
        model.warmup(epoch)
        if device != "cpu":
            model = DataParallel(model)
        model.to(device)
        model.train()

        for batch_i, batch in enumerate(tqdm(loader)):
            1

        self.logger.info(f"trn_loss: {1}")

        if self.device != "cpu":
            model = model.module
        model.to("cpu")

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
        dataset = self.dataset_factory.create(df=df)
        sampler = self.sampler_factory.create(sampler_type=sampler_type)
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

    def _build_model(self) -> Tuple[Module, Optimizer, _LRScheduler]:
        model = self.model_factory.create()
        optimizer = self.optimizer_factory.create(model=model)
        scheduler = self.scheduler_factory.create(model=model)
        return model, optimizer, scheduler

    @class_dec_timer(unit="m")
    def _valid(
        self, fold_num: int, model: Module, loader: DataLoader, fobj: _Loss
    ) -> None:
        1
