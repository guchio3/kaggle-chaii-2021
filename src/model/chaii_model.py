from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Conv1d, Dropout, Softmax
from torch.nn.modules.loss import _Loss

from src.log import myLogger
from src.loss import lovasz_hinge
from src.model.model import Model


class ChaiiXLMRBModel1(Model):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        warmup_epoch: int,
        max_grad_norm: Optional[float],
        start_loss_weight: float,
        end_loss_weight: float,
        segmentation_loss_weight: float,
        logger: myLogger,
    ) -> None:
        if segmentation_loss_weight != 0.0:
            raise Exception(
                f"segmentation_loss_weight for {self.__class__.__name__} should be 0,"
                f"{segmentation_loss_weight} was found."
            )

        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model_type="model",
            warmup_keys=[
                "pooler",
                "classifier_dropout",
                "classifier_conv_start",
                "classifier_conv_end",
            ],
            warmup_epoch=warmup_epoch,
            max_grad_norm=max_grad_norm,
            start_loss_weight=start_loss_weight,
            end_loss_weight=end_loss_weight,
            segmentation_loss_weight=segmentation_loss_weight,
            logger=logger,
        )
        self.classifier_dropout = Dropout(0.2)
        self.classifier_conv_start = Conv1d(self.model.pooler.dense.out_features, 1, 1)
        self.classifier_conv_end = Conv1d(self.model.pooler.dense.out_features, 1, 1)
        # self.add_module("conv_output_start", self.classifier_conv_start)
        # self.add_module("conv_output_end", self.classifier_conv_end)

    def forward(
        self, input_ids: Tensor, attention_masks: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_masks,)
        output = outputs[0]
        output = torch.transpose(output, 1, 2)
        output = self.classifier_dropout(output)
        start_logits = self.classifier_conv_start(output).squeeze()
        end_logits = self.classifier_conv_end(output).squeeze()

        return start_logits, end_logits, torch.zeros(start_logits.shape)

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
        loss = self.start_loss_weight * fobj(start_logits, start_positions)
        loss += self.end_loss_weight * fobj(end_logits, end_positions)
        return loss


class ChaiiQAXLMRBModel1(Model):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        warmup_epoch: int,
        max_grad_norm: Optional[float],
        start_loss_weight: float,
        end_loss_weight: float,
        segmentation_loss_weight: float,
        logger: myLogger,
    ) -> None:
        if segmentation_loss_weight != 0:
            raise Exception(
                f"segmentation_loss_weight for {self.__class__.__name__} should be 0,"
                f"{segmentation_loss_weight} was found."
            )

        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model_type="qa_model",
            warmup_keys=[],
            warmup_epoch=warmup_epoch,
            max_grad_norm=max_grad_norm,
            start_loss_weight=start_loss_weight,
            end_loss_weight=end_loss_weight,
            segmentation_loss_weight=segmentation_loss_weight,
            logger=logger,
        )

    def forward(
        self, input_ids: Tensor, attention_masks: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        output = self.model(input_ids=input_ids, attention_mask=attention_masks,)
        start_logits = output.start_logits
        end_logits = output.end_logits

        return start_logits, end_logits, torch.zeros(start_logits.shape)

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
        loss = self.start_loss_weight * fobj(start_logits, start_positions)
        loss += self.end_loss_weight * fobj(end_logits, end_positions)
        return loss


class ChaiiQASegXLMRBModel1(Model):
    def __init__(
        self,
        pretrained_model_name_or_path: str,
        warmup_epoch: int,
        max_grad_norm: Optional[float],
        start_loss_weight: float,
        end_loss_weight: float,
        segmentation_loss_weight: float,
        logger: myLogger,
    ) -> None:
        if segmentation_loss_weight == 0:
            raise Exception(
                f"segmentation_loss_weight for {self.__class__.__name__} should not be 0."
            )

        super().__init__(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            model_type="qa_model",
            warmup_keys=[],
            warmup_epoch=warmup_epoch,
            max_grad_norm=max_grad_norm,
            start_loss_weight=start_loss_weight,
            end_loss_weight=end_loss_weight,
            segmentation_loss_weight=segmentation_loss_weight,
            logger=logger,
        )
        self.softmax = Softmax(dim=1)

    def forward(
        self, input_ids: Tensor, attention_masks: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        output = self.model(input_ids=input_ids, attention_mask=attention_masks,)
        start_logits = output.start_logits
        end_logits = output.end_logits

        return start_logits, end_logits, torch.zeros(start_logits.shape)

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
        loss = self.start_loss_weight * fobj(start_logits, start_positions)
        loss += self.end_loss_weight * fobj(end_logits, end_positions)

        start_cumsumed = self.softmax(start_logits).cumsum(dim=1)
        end_cumsumed = (
            self.softmax(end_logits).flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
        )

        segementation_pred_proba = start_cumsumed * end_cumsumed
        segmentation_logits = torch.log(
            (segementation_pred_proba + 1e-10) / (1 - segementation_pred_proba + 1e-10)
        )
        loss += self.segmentation_loss_weight * lovasz_hinge(
            segmentation_logits, segmentation_positions, ignore=-1
        )

        return loss
