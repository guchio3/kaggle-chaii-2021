import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from pandas import DataFrame
from torch.nn import DataParallel, Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.checkpoint.checkpoint import Checkpoint
from src.dataset.factory import DatasetFactory
from src.fobj.factory import FobjFactory
from src.log import myLogger
from src.model.factory import ModelFactory
from src.model.model import Model
from src.optimizer.factory import OptimizerFactory
from src.pipeline.pipeline import Pipeline
from src.preprocessor.factory import PreprocessorFactory
from src.repository.checkpoint_repository import CheckpointRepository
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
        debug: bool,
        logger: myLogger,
    ) -> None:
        super().__init__("train_pred", exp_id, logger)
        self.config = config
        self.device = device
        self.debug = debug

        self.data_repository = DataRepository(logger=logger)
        self.checkpoint_repository = CheckpointRepository(logger=logger)

        self.num_epochs = config["num_epochs"]

        self.preprocessor_factory = PreprocessorFactory(
            **config["preprocessor"], logger=logger
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

    @class_dec_timer(unit="m")
    def run(self, mode: str) -> None:
        if mode == "train":
            self._train()
        elif mode == "pred":
            self._pred()
        else:
            raise NotImplementedError(f"mode {mode} is not supported.")

    @class_dec_timer(unit="m")
    def _train(self) -> None:
        trn_df = self.data_repository.load_train_df()
        preprocessor = self.preprocessor_factory.create(
            data_repository=self.data_repository
        )
        trn_df = preprocessor(trn_df)

        splitter = self.splitter_factory.create()
        folds = splitter.split(trn_df["id"], trn_df["language"], groups=None)

        for fold, (trn_idx, val_idx) in enumerate(folds):
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
                    fold=fold,
                    epoch=epoch,
                    accum_mod=1,
                    loader=trn_loader,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    fobjs={"fobj": fobj, "fobj_segmentation": None},
                )
                checkpoint = Checkpoint(exp_id=self.exp_id, fold=fold, epoch=epoch)
                checkpoint.set_model(model=model)
                checkpoint.set_optimizer(optimizer=optimizer)
                checkpoint.set_scheduler(scheduler=scheduler)
                self._valid(fold=fold, model=model, loader=val_loader, fobj=fobj)
                self.checkpoint_repository.save(
                    fold=fold,
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
        fold: int,
        epoch: int,
        accum_mod: int,
        loader: DataLoader,
        model: Model,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        fobjs: Dict[str, Optional[_Loss]],
    ) -> None:
        # init for train
        model.warmup(epoch)
        if device != "cpu":
            model = DataParallel(model)
        self._to_device(device=self.device, model=model, optimizer=optimizer)
        model.train()

        running_loss = 0.0
        for batch_i, batch in enumerate(tqdm(loader)):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            start_position = batch["start_position"].to(self.device)
            end_position = batch["end_position"].to(self.device)
            segmentation_positions = batch["segmentation_positions"].to(self.device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            loss = model.calc_loss(
                logits=logits,
                fobjs=fobjs,
                start_position=start_position,
                end_position=end_position,
                segmentation_positions=segmentation_positions,
            )
            loss.backward()
            running_loss += loss.item()
            if (batch_i + 1) % accum_mod == 0:
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()

        running_loss /= len(loader)
        self.logger.info(f"fold: {fold} / epoch: {epoch} / trn_loss: {running_loss}")
        self.logger.wdb_log({"epoch": epoch, f"train/fold_{fold}_loss": running_loss})

        if device != "cpu":
            model = model.module
        self._to_device(device="cpu", model=model, optimizer=optimizer)

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

    def _build_model(self) -> Tuple[Model, Optimizer, _LRScheduler]:
        model = self.model_factory.create()
        optimizer = self.optimizer_factory.create(model=model)
        scheduler = self.scheduler_factory.create(optimizer=optimizer)
        return model, optimizer, scheduler

    @class_dec_timer(unit="m")
    def _valid(self, fold: int, model: Module, loader: DataLoader, fobj: _Loss) -> None:
        1

    def _to_device(self, device: str, model: Model, optimizer: Optimizer) -> None:
        model.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor):
                    state[k] = v.to(device)
