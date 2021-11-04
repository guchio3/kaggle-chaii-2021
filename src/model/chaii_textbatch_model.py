from typing import Optional, Tuple

import numpy as np
import torch
from sklearn import metrics
from torch import Tensor
from torch.nn import Linear, Sigmoid
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm

from src.checkpoint.checkpoint import Checkpoint
from src.log import myLogger
from src.model.model import Model
from src.timer import class_dec_timer


class ChaiiTextBatchXLMRBModel1(Model):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        warmup_epoch: int,
        max_grad_norm: Optional[float],
        logger: myLogger,
        **kwargs,
    ) -> None:
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model_type="model",
            warmup_keys=[],
            warmup_epoch=warmup_epoch,
            max_grad_norm=max_grad_norm,
            start_loss_weight=0.0,
            end_loss_weight=0.0,
            segmentation_loss_weight=0.0,
            logger=logger,
        )
        self.classifier = Linear(self.model.pooler.dense.out_features, 1)

    def forward(
        self, input_ids: Tensor, attention_masks: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError()

    def textbatch_forward(self, input_ids: Tensor, attention_masks: Tensor) -> Tensor:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_masks,)
        output = outputs[0]
        logits = self.classifier(output)
        return logits[:, 0].squeeze()

    def calc_loss(
        self,
        start_logits: Tensor,
        end_logits: Tensor,
        segmentation_logits: Tensor,
        start_positions: Tensor,
        end_positions: Tensor,
        segmentation_positions: Tensor,
        fobj: Optional[_Loss],
        segmentation_fobj: Optional[_Loss],
    ) -> Tensor:
        raise NotImplementedError()

    def calc_textbatch_loss(
        self,
        logits: Tensor,
        is_contain_answer_texts: Tensor,
        fobj: Optional[_Loss],
    ) -> Tensor:
        if fobj is None:
            raise Exception("plz set fobj.")
        loss = fobj(logits, is_contain_answer_texts)
        return loss

    @class_dec_timer(unit="m")
    def textbatch_train_one_epoch(
        self,
        device: str,
        fold: int,
        epoch: int,
        accum_mod: int,
        loader: DataLoader,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        fobj: Optional[_Loss],
    ) -> None:
        # init for train
        self.warmup(epoch)
        # if device != "cpu":
        #     model = DataParallel(model)
        self._to_device(device=device, optimizer=optimizer)
        self.train()
        self.zero_grad()

        running_loss = 0.0
        for batch_i, batch in enumerate(tqdm(loader)):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_mask"].to(device)
            is_contain_answer_texts = batch["is_contain_answer_text"].to(device)

            logits = self.textbatch_forward(
                input_ids=input_ids, attention_masks=attention_masks,
            )
            loss = self.calc_textbatch_loss(
                logits=logits,
                is_contain_answer_texts=is_contain_answer_texts,
                fobj=fobj,
            )
            running_loss += loss.item()  # runnig_loss uses non scaled loss
            loss = loss / accum_mod
            loss.backward()
            loss.detach()
            if (batch_i + 1) % accum_mod == 0:
                self.clip_grad_norm()
                optimizer.step()
                # optimizer.zero_grad()
                self.zero_grad()

        scheduler.step()

        trn_loss = running_loss / len(loader)
        self.logger.info(f"fold: {fold} / epoch: {epoch} / trn_loss: {trn_loss:.4f}")
        self.logger.wdb_log({"epoch": epoch, f"train/fold_{fold}_loss": trn_loss})

        # if device != "cpu":
        #     model = model.module
        self._to_device(device="cpu", optimizer=optimizer)

    @class_dec_timer(unit="m")
    def textbatch_valid(
        self,
        device: str,
        fold: int,
        epoch: int,
        loader: DataLoader,
        fobj: Optional[_Loss],
        checkpoint: Checkpoint,
    ) -> None:
        # if device != "cpu":
        #     model = DataParallel(model)
        self.to(device)
        self.eval()

        all_is_contain_answer_texts = []
        all_sigmoided_logits = []
        m = Sigmoid()
        with torch.no_grad():
            running_loss = 0.0
            for _, batch in enumerate(tqdm(loader)):
                input_ids = batch["input_ids"].to(device)
                attention_masks = batch["attention_mask"].to(device)
                is_contain_answer_texts = batch["is_contain_answer_text"].to(device)
                ids = batch["id"]
                input_ids = batch["input_ids"]
                attention_masks = batch["attention_mask"]

                logits = self.textbatch_forward(
                    input_ids=input_ids, attention_masks=attention_masks,
                )
                if logits.dim() == 1:
                    self.logger.info(
                        "fix the shape of logits because it contains just one elem."
                    )
                    logits = logits.reshape(1, -1)

                loss = self.calc_textbatch_loss(
                    logits=logits,
                    is_contain_answer_texts=is_contain_answer_texts,
                    fobj=fobj,
                )
                sigmoided_logits = m(logits)

                logits = logits.to("cpu")
                sigmoided_logits = sigmoided_logits.to("cpu").numpy()
                is_contain_answer_texts = is_contain_answer_texts.to("cpu").numpy()

                all_is_contain_answer_texts.append(is_contain_answer_texts)
                all_sigmoided_logits.append(sigmoided_logits)

                checkpoint.extend_str_list_val_info(key="val_ids", val_info=ids)
                checkpoint.extend_tensor_val_info(
                    key="val_start_logits", val_info=logits
                )
                checkpoint.extend_tensor_val_info(key="val_end_logits", val_info=logits)
                checkpoint.extend_tensor_val_info(
                    key="val_segmentation_logits", val_info=logits
                )

                running_loss += loss.item()

            final_all_is_contain_answer_texts = np.concatenate(
                all_is_contain_answer_texts
            )
            final_all_sigmoided_logits = np.concatenate(all_sigmoided_logits)

            val_loss = running_loss / len(loader)
            val_auc = metrics.roc_auc_score(
                y_true=final_all_is_contain_answer_texts,
                y_score=final_all_sigmoided_logits,
            )
            checkpoint.val_pospro_ids = checkpoint.val_ids
            checkpoint.val_pospro_answer_texts = checkpoint.val_ids
            checkpoint.val_pospro_answer_preds = checkpoint.val_ids
            checkpoint.val_loss = val_loss
            checkpoint.val_jaccard = val_auc

            self.logger.info(
                f"fold: {fold} / epoch: {epoch} / val_loss: {val_loss:.4f} / val_jaccard: {val_auc:.4f}"
            )
            self.logger.wdb_log(
                {
                    "epoch": epoch,
                    f"valid/fold_{fold}_loss": val_loss,
                    f"valid/fold_{fold}_jaccard": val_auc,
                }
            )

        # if device != "cpu":
        #     model = model.module
        self.to("cpu")
