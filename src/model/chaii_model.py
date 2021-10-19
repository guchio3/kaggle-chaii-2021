from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Conv1d, Dropout
from torch.nn.modules.loss import _Loss

from src.log import myLogger
from src.model.model import Model


class ChaiiXLMRBModel1(Model):
    def __init__(
        self, pretrained_model_name_or_path: str, warmup_epoch: int, logger: myLogger
    ) -> None:
        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            warmup_keys=[
                "conv_output",
                "roberta.pooler.dense.weight",
                "roberta.pooler.dense.bias",
            ],
            warmup_epoch=warmup_epoch,
            logger=logger,
        )
        self.dropout = Dropout(0.2)
        self.classifier_conv_start = Conv1d(self.model.pooler.dense.out_features, 1, 1)
        self.classifier_conv_end = Conv1d(self.model.pooler.dense.out_features, 1, 1)
        self.add_module("conv_output_start", self.classifier_conv_start)
        self.add_module("conv_output_end", self.classifier_conv_end)

    def forward(
        self, input_ids: Tensor, attention_masks: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_masks,)
        output = outputs[0]
        output = torch.transpose(output, 1, 2)
        output = self.dropout(output)
        start_logits = self.classifier_conv_start(output).squeeze()
        end_logits = self.classifier_conv_end(output).squeeze()

        return start_logits, end_logits, Tensor()

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
        if fobj is None:
            raise Exception("plz set fobj.")
        loss = fobj(start_logits, start_positions)
        loss += fobj(end_logits, end_positions)
        return loss
