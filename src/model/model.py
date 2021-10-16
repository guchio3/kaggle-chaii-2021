from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Tuple

from torch import Tensor
from torch.nn import Module
from torch.nn.modules.loss import _Loss
from transformers import AutoModel

from src.log import myLogger


class Model(Module, metaclass=ABCMeta):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        warmup_key: str,
        warmup_epoch: int,
        logger: myLogger,
    ) -> None:
        super().__init__()
        if isinstance(pretrained_model_name_or_path, str):
            self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        else:
            # for sub
            self.model = AutoModel(pretrained_model_name_or_path)
        self.warmup_key = warmup_key
        self.warmup_epoch = warmup_epoch
        self.logger = logger

    @abstractmethod
    def forward(
        self, input_ids: Tensor, attention_masks: Tensor
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        raise NotImplementedError()

    def warmup(self, epoch: int) -> None:
        if epoch == 0:
            for name, child in self.named_children():
                if self.warmup_key in name:
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
        segmentation_fobj: Optional[_Loss]
    ) -> Tensor:
        raise NotImplementedError()

    def resize_token_embeddings(self, token_num: int) -> None:
        self.model.resize_token_embeddings(token_num)
