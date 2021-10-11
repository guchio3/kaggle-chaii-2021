from typing import Dict, List, Optional

import torch
from torch import Tensor
from torch.nn import Conv1d, Dropout
from torch.nn.modules.loss import _Loss

from src.log import myLogger
from src.model.model import Model


class ChaiiXLMRBModel1(Model):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        warmup_epoch: int,
        logger: myLogger,
    ) -> None:
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            warmup_key="conv_output",
            warmup_epoch=warmup_epoch,
            logger=logger,
        )
        self.dropout = Dropout(0.2)
        self.classifier_conv_start = Conv1d(self.model.pooler.dense.out_features, 1, 1)
        self.classifier_conv_end = Conv1d(self.model.pooler.dense.out_features, 1, 1)
        self.add_module("conv_output_start", self.classifier_conv_start)
        self.add_module("conv_output_end", self.classifier_conv_end)

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> List[Tensor]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        output = outputs[0]
        output = torch.transpose(output, 1, 2)
        output = self.dropout(output)
        logits_start = self.classifier_conv_start(output).squeeze()
        logits_end = self.classifier_conv_end(output).squeeze()

        return [logits_start, logits_end]

    def calc_loss(
        self,
        logits: List[Tensor],
        fobjs: Dict[str, Optional[_Loss]],
        start_position: Tensor,
        end_position: Tensor,
        **kwargs
    ) -> Tensor:
        logits_start, logits_end = logits
        fobj = fobjs["fobj"]

        if fobj is None:
            raise Exception("plz set fobj.")
        loss = fobj(logits_start, start_position)
        loss += fobj(logits_end, end_position)

        return loss
