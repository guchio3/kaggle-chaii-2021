from typing import Optional

from src.factory import Factory
from src.log import myLogger
from src.model.chaii_model import (ChaiiQASegXLMRBModel1, ChaiiQAXLMRBModel1,
                                   ChaiiXLMRBModel1)
from src.model.chaii_textbatch_model import ChaiiTextBatchXLMRBModel1
from src.model.model import Model


class ModelFactory(Factory[Model]):
    def __init__(
        self,
        model_type: str,
        pretrained_model_name_or_path: str,
        warmup_epoch: int,
        max_grad_norm: Optional[float],
        start_loss_weight: float,
        end_loss_weight: float,
        segmentation_loss_weight: float,
        logger: myLogger,
    ):
        super().__init__(
            model_type=model_type,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            warmup_epoch=warmup_epoch,
            max_grad_norm=max_grad_norm,
            start_loss_weight=start_loss_weight,
            end_loss_weight=end_loss_weight,
            segmentation_loss_weight=segmentation_loss_weight,
            logger=logger,
        )

    def _create(self,) -> Model:
        if self.model_type == "chaii-xlmrb-1":
            model = ChaiiXLMRBModel1(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                warmup_epoch=self.warmup_epoch,
                max_grad_norm=self.max_grad_norm,
                start_loss_weight=self.start_loss_weight,
                end_loss_weight=self.end_loss_weight,
                segmentation_loss_weight=self.segmentation_loss_weight,
                logger=self.logger,
            )
        elif self.model_type == "chaii-qa-xlmrb-1":
            model = ChaiiQAXLMRBModel1(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                warmup_epoch=self.warmup_epoch,
                max_grad_norm=self.max_grad_norm,
                start_loss_weight=self.start_loss_weight,
                end_loss_weight=self.end_loss_weight,
                segmentation_loss_weight=self.segmentation_loss_weight,
                logger=self.logger,
            )
        elif self.model_type == "chaii-qa-seg-xlmrb-1":
            model = ChaiiQASegXLMRBModel1(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                warmup_epoch=self.warmup_epoch,
                max_grad_norm=self.max_grad_norm,
                start_loss_weight=self.start_loss_weight,
                end_loss_weight=self.end_loss_weight,
                segmentation_loss_weight=self.segmentation_loss_weight,
                logger=self.logger,
            )
        elif self.model_type == "chaii-text-batch-xlmrb-1":
            model = ChaiiTextBatchXLMRBModel1(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                warmup_epoch=self.warmup_epoch,
                max_grad_norm=self.max_grad_norm,
                logger=self.logger,
            )
        else:
            raise NotImplementedError(f"model_type {self.model_type} is not supported.")
        return model
