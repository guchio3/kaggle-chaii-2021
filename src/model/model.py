from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional

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
        logits: List[Tensor],
        fobjs: Dict[str, Optional[_Loss]],
        start_position: Tensor,
        end_position: Tensor,
        segmentation_positions: Tensor,
    ) -> Tensor:
        raise NotImplementedError()

    def resize_token_embeddings(self, token_num: int) -> None:
        self.model.resize_token_embeddings(token_num)
