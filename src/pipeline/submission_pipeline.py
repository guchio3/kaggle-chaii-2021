import itertools
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.nn.modules.loss import _Loss
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from src.checkpoint.checkpoint import Checkpoint
from src.config import ConfigLoader
from src.dataset.factory import DatasetFactory
from src.log import myLogger
from src.metrics.jaccard import calc_jaccard_mean
from src.model.factory import ModelFactory
from src.model.model import Model
from src.pipeline.pipeline import Pipeline
from src.postprocessor.factory import PostprocessorFactory
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
        config_local_root_path: str,
        debug: bool,
        logger: myLogger,
    ) -> None:
        super().__init__("train_pred", exp_id, logger)
        self.config = config
        self.device = device
        self.debug = debug

        self.data_repository = DataRepository(origin_root_path=data_origin_root_path, logger=logger)
        self.config_loader = ConfigLoader(local_root_path=config_local_root_path)

        self.tst_batch_size = config["tst_batch_size"]
        self. = config[""]

        self.preprocessor_factory = PreprocessorFactory(
            **config["preprocessor"], debug=debug, logger=logger
        )
        self.postprocessor_factory = PostprocessorFactory(
            **config["postprocessor"], logger=logger
        )
        self.dataset_factory = DatasetFactory(**config["dataset"], logger=logger)
        self.sampler_factory = SamplerFactory(**config["sampler"], logger=logger)
        self.model_factory = ModelFactory(**config["model"], logger=logger)

    def run(self) -> None:
        self._create_submission()

    @class_dec_timer(unit="m")
    def _create_submission(self) -> None:
        tst_df = self.data_repository.load_test_df()
        preprocessor = self.preprocessor_factory.create(
            data_repository=self.data_repository
        )
        preprocessed_trn_df = preprocessor(
            df=tst_df,
            dataset_name="test",
            enforce_preprocess=False,
            is_test=True,
        )

        for exp_id in self.ensemble_exp_ids:

        for fold, (trn_idx, val_idx) in enumerate(folds):
            if fold not in self.train_folds:
                self.logger.info("skip fold {fold} because it's not in train_folds.")
                continue
            # fold data
            trn_ids = trn_df.iloc[trn_idx]["id"].tolist()
            fold_trn_df = preprocessed_trn_df.query(f"id in {trn_ids}")
            fold_trn_df = pd.concat(
                [fold_trn_df, preprocessed_booster_train_df], axis=0
            ).reset_index(drop=True)
            trn_loader = self._build_loader(
                df=fold_trn_df,
                sampler_type=self.config["sampler"]["trn_sampler_type"],
                batch_size=self.trn_batch_size,
                drop_last=True,
                debug=self.debug,
            )
            val_ids = trn_df.iloc[val_idx]["id"].tolist()
            fold_val_df = preprocessed_trn_df.query(f"id in {val_ids}")
            val_loader = self._build_loader(
                df=fold_val_df,
                sampler_type=self.config["sampler"]["val_sampler_type"],
                batch_size=self.val_batch_size,
                drop_last=False,
                debug=self.debug,
            )

            # fold model
            model, optimizer, scheduler = self._build_model()
            fobj = self.fobj_factory.create()

            val_jaccards = []
            for epoch in range(self.num_epochs):
                self._train_one_epoch(
                    device=self.device,
                    fold=fold,
                    epoch=epoch,
                    accum_mod=self.accum_mod,
                    loader=trn_loader,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    fobj=fobj,
                    segmentation_fobj=None,
                )
                checkpoint = Checkpoint(exp_id=self.exp_id, fold=fold, epoch=epoch)
                checkpoint.set_model(model=model)
                checkpoint.set_optimizer(optimizer=optimizer)
                checkpoint.set_scheduler(scheduler=scheduler)
                self._valid(
                    device=self.device,
                    fold=fold,
                    epoch=epoch,
                    accum_mod=self.accum_mod,
                    model=model,
                    loader=val_loader,
                    fobj=fobj,
                    segmentation_fobj=None,
                    checkpoint=checkpoint,
                )
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

    @class_dec_timer(unit="m")
    def _valid(
        self,
        device: str,
        fold: int,
        epoch: int,
        model: Model,
        loader: DataLoader,
        fobj: Optional[_Loss],
        segmentation_fobj: Optional[_Loss],
        checkpoint: Checkpoint,
    ) -> None:
        model.to(device)
        model.eval()

        with torch.no_grad():
            running_loss = 0.0
            all_ids = []
            all_contexts = []
            all_answer_texts = []
            all_offset_mappings = []
            all_start_logits = []
            all_end_logits = []
            all_segmentation_logits = []
            for _, batch in enumerate(tqdm(loader)):
                ids = batch["id"]
                contexts = batch["context"]
                answer_text = batch["answer_text"]
                offset_mappings = batch["offset_mapping"]
                input_ids = batch["input_ids"].to(device)
                attention_masks = batch["attention_mask"].to(device)
                start_positions = batch["start_position"].to(device)
                end_positions = batch["end_position"].to(device)
                segmentation_positions = batch["segmentation_position"].to(device)

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

                loss = model.calc_loss(
                    start_logits=start_logits,
                    end_logits=end_logits,
                    segmentation_logits=segmentation_logits,
                    start_positions=start_positions,
                    end_positions=end_positions,
                    segmentation_positions=segmentation_positions,
                    fobj=fobj,
                    segmentation_fobj=segmentation_fobj,
                )
                loss = loss

                start_logits.to("cpu")
                end_logits.to("cpu")
                segmentation_logits.to("cpu")

                all_ids.append(ids)
                all_contexts.append(contexts)
                all_answer_texts.append(answer_text)
                all_offset_mappings.append(offset_mappings)
                all_start_logits.append(start_logits)
                all_end_logits.append(end_logits)
                all_segmentation_logits.append(segmentation_logits)

                checkpoint.extend_str_list_val_info(key="val_ids", val_info=ids)
                checkpoint.extend_tensor_val_info(
                    key="val_start_logits", val_info=start_logits
                )
                checkpoint.extend_tensor_val_info(
                    key="val_end_logits", val_info=end_logits
                )
                checkpoint.extend_tensor_val_info(
                    key="val_segmentation_logits", val_info=segmentation_logits
                )

                running_loss += loss.item()

            final_all_ids = list(itertools.chain.from_iterable(all_ids))
            final_all_contexts = list(itertools.chain.from_iterable(all_contexts))
            final_all_answer_texts = list(
                itertools.chain.from_iterable(all_answer_texts)
            )
            final_all_offset_mappings = list(
                itertools.chain.from_iterable(all_offset_mappings)
            )
            final_all_start_logits = torch.cat(all_start_logits).to("cpu")
            final_all_end_logits = torch.cat(all_end_logits).to("cpu")
            final_all_segmentation_logits = torch.cat(all_segmentation_logits).to("cpu")

            val_loss = running_loss / len(loader)
            postprocessor = self.postprocessor_factory.create()
            pospro_ids, pospro_answer_texts, pospro_answer_preds = postprocessor(
                ids=final_all_ids,
                contexts=final_all_contexts,
                answer_texts=final_all_answer_texts,
                offset_mappings=final_all_offset_mappings,
                start_logits=final_all_start_logits,
                end_logits=final_all_end_logits,
                segmentation_logits=final_all_segmentation_logits,
            )
            val_jaccard = calc_jaccard_mean(
                text_trues=pospro_answer_texts, text_preds=pospro_answer_preds
            )

            checkpoint.val_pospro_ids = pospro_ids
            checkpoint.val_pospro_answer_texts = pospro_answer_texts
            checkpoint.val_pospro_answer_preds = pospro_answer_preds
            checkpoint.val_loss = val_loss
            checkpoint.val_jaccard = val_jaccard

            self.logger.info(
                f"fold: {fold} / epoch: {epoch} / val_loss: {val_loss:.4f} / val_jaccard: {val_jaccard:.4f}"
            )
            self.logger.wdb_log(
                {
                    "epoch": epoch,
                    f"valid/fold_{fold}_loss": val_loss,
                    f"valid/fold_{fold}_jaccard": val_jaccard,
                }
            )

        model.to("cpu")
