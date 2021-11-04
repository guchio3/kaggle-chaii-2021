import gc
import itertools
from abc import ABCMeta, abstractmethod
from typing import Iterator, List, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForQuestionAnswering

from src.checkpoint.checkpoint import Checkpoint
from src.log import myLogger
from src.metrics.jaccard import calc_jaccard_mean
from src.postprocessor.postprocessor import Postprocessor
from src.prediction.prediction_result import PredictionResult
from src.timer import class_dec_timer


class Model(Module, metaclass=ABCMeta):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        model_type: str,
        warmup_keys: List[str],
        warmup_epoch: int,
        max_grad_norm: Optional[float],
        start_loss_weight: float,
        end_loss_weight: float,
        segmentation_loss_weight: float,
        logger: myLogger,
    ) -> None:
        super().__init__()
        if isinstance(pretrained_model_name_or_path, str):
            if model_type == "model":
                self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
            elif model_type == "qa_model":
                self.model = AutoModelForQuestionAnswering.from_pretrained(
                    pretrained_model_name_or_path
                )
            else:
                raise Exception("model_type {model_type} is not supported.")
        else:
            # for sub
            self.model = AutoModel(pretrained_model_name_or_path)
            if model_type == "model":
                self.model = AutoModel(pretrained_model_name_or_path)
            elif model_type == "qa_model":
                self.model = AutoModelForQuestionAnswering(
                    pretrained_model_name_or_path
                )
            else:
                raise Exception("model_type {model_type} is not supported.")

        self.warmup_keys = warmup_keys
        self.warmup_epoch = warmup_epoch
        self.max_grad_norm = max_grad_norm

        self.start_loss_weight = start_loss_weight
        self.end_loss_weight = end_loss_weight
        self.segmentation_loss_weight = segmentation_loss_weight

        self.logger = logger

    @abstractmethod
    def forward(
        self, input_ids: Tensor, attention_masks: Tensor
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        raise NotImplementedError()

    def warmup(self, epoch: int) -> None:
        if self.warmup_epoch != 0 and epoch == 0:
            for name, child in self.named_children():
                is_key_in = False
                for warmup_key in self.warmup_keys:
                    if warmup_key in name:
                        is_key_in = True
                        break
                if is_key_in:
                    self.logger.info(name + " is unfrozen")
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    self.logger.info(name + " is frozen")
                    for param in child.parameters():
                        param.requires_grad = False
        if epoch == self.warmup_epoch:
            self.logger.info("Turn on all the layers")
            # for name, child in model.named_children():
            for name, child in self.named_children():
                for param in child.parameters():
                    param.requires_grad = True

    def clip_grad_norm(self) -> None:
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(
                self.parameters(),
                self.max_grad_norm,
            )

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        named_children = {
            k: v for k, v in super().named_children() if k != "model"
        }  # remove model to use model's each children
        named_children.update(self.model.named_children())
        for name, child in named_children.items():
            yield name, child

    @abstractmethod
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

    def resize_token_embeddings(self, token_num: int) -> None:
        self.model.resize_token_embeddings(token_num)

    @class_dec_timer(unit="m")
    def train_one_epoch(
        self,
        device: str,
        fold: int,
        epoch: int,
        accum_mod: int,
        loader: DataLoader,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        fobj: Optional[_Loss],
        segmentation_fobj: Optional[_Loss],
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
            start_positions = batch["start_position"].to(device)
            end_positions = batch["end_position"].to(device)
            segmentation_positions = batch["segmentation_position"].to(device)

            start_logits, end_logits, segmentation_logits = self(
                input_ids=input_ids,
                attention_masks=attention_masks,
            )
            loss = self.calc_loss(
                start_logits=start_logits,
                end_logits=end_logits,
                segmentation_logits=segmentation_logits,
                start_positions=start_positions,
                end_positions=end_positions,
                segmentation_positions=segmentation_positions,
                fobj=fobj,
                segmentation_fobj=segmentation_fobj,
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
    def valid(
        self,
        device: str,
        fold: int,
        epoch: int,
        loader: DataLoader,
        fobj: Optional[_Loss],
        segmentation_fobj: Optional[_Loss],
        postprocessor: Postprocessor,
        checkpoint: Checkpoint,
    ) -> None:
        # if device != "cpu":
        #     model = DataParallel(model)
        self.to(device)
        self.eval()

        with torch.no_grad():
            running_loss = 0.0
            all_contexts = []
            all_answer_texts = []
            all_offset_mappings = []
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

                start_logits, end_logits, segmentation_logits = self(
                    input_ids=input_ids,
                    attention_masks=attention_masks,
                )
                if start_logits.dim() == 1:
                    self.logger.info(
                        "fix the shape of logits because it contains just one elem."
                    )
                    start_logits = start_logits.reshape(1, -1)
                    end_logits = end_logits.reshape(1, -1)
                    segmentation_logits = segmentation_logits.reshape(1, -1)

                loss = self.calc_loss(
                    start_logits=start_logits,
                    end_logits=end_logits,
                    segmentation_logits=segmentation_logits,
                    start_positions=start_positions,
                    end_positions=end_positions,
                    segmentation_positions=segmentation_positions,
                    fobj=fobj,
                    segmentation_fobj=segmentation_fobj,
                )

                start_logits = start_logits.to("cpu")
                end_logits = end_logits.to("cpu")
                segmentation_logits = segmentation_logits.to("cpu")

                all_contexts.append(contexts)
                all_answer_texts.append(answer_text)
                all_offset_mappings.append(offset_mappings)

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

            final_all_contexts = list(itertools.chain.from_iterable(all_contexts))
            final_all_answer_texts = list(
                itertools.chain.from_iterable(all_answer_texts)
            )
            final_all_offset_mappings = list(
                itertools.chain.from_iterable(all_offset_mappings)
            )

            val_loss = running_loss / len(loader)
            pospro_ids, pospro_answer_texts, pospro_answer_preds = postprocessor(
                ids=checkpoint.val_ids,
                contexts=final_all_contexts,
                answer_texts=final_all_answer_texts,
                offset_mappings=final_all_offset_mappings,
                start_logits=checkpoint.val_start_logits,
                end_logits=checkpoint.val_end_logits,
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

        # if device != "cpu":
        #     model = model.module
        self.to("cpu")

    @class_dec_timer(unit="m")
    def predict(
        self,
        device: str,
        ensemble_weight: float,
        loader: DataLoader,
    ) -> PredictionResult:
        self.to(device)
        self.eval()

        prediction_result = PredictionResult(ensemble_weight=ensemble_weight)
        with torch.no_grad():
            for _, batch in enumerate(tqdm(loader)):
                ids = batch["id"]
                offset_mappings = batch["offset_mapping"]
                input_ids = batch["input_ids"].to(device)
                attention_masks = batch["attention_mask"].to(device)

                start_logits, end_logits, _segmentation_logits = self(
                    input_ids=input_ids,
                    attention_masks=attention_masks,
                )
                if start_logits.dim() == 1:
                    self.logger.info(
                        "fix the shape of logits because it contains just one elem."
                    )
                    start_logits = start_logits.reshape(1, -1)
                    end_logits = end_logits.reshape(1, -1)
                    # segmentation_logits = segmentation_logits.reshape(1, -1)

                prediction_result.extend_by_value_list(key="ids", value_list=ids)
                prediction_result.extend_by_tensor(
                    key="offset_mappings", val_info=offset_mappings
                )
                prediction_result.extend_by_tensor(
                    key="start_logits", val_info=start_logits.to("cpu")
                )
                prediction_result.extend_by_tensor(
                    key="end_logits", val_info=end_logits.to("cpu")
                )

                del ids
                del offset_mappings
                del input_ids
                del attention_masks
                del start_logits
                del end_logits
                del _segmentation_logits
                gc.collect()

        self.to("cpu")
        torch.cuda.empty_cache()

        return prediction_result

    def _to_device(self, device: str, optimizer: Optimizer) -> None:
        self.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, Tensor):
                    state[k] = v.to(device)
