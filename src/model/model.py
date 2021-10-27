from abc import ABCMeta, abstractmethod
from typing import Iterator, List, Optional, Tuple

from torch import Tensor, nn
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from transformers import AutoModel, AutoModelForQuestionAnswering

from src.log import myLogger


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
                self.parameters(), self.max_grad_norm,
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
