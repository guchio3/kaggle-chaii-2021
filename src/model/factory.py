from src.factory import Factory
from src.log import myLogger
from src.model.chaii_model import ChaiiXLMRBModel1
from src.model.model import Model


class ModelFactory(Factory[Model]):
    def __init__(
        self,
        model_type: str,
        pretrained_model_name_or_path: str,
        warmup_epoch: int,
        logger: myLogger,
    ):
        super().__init__(
            model_type=model_type,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            warmup_epoch=warmup_epoch,
            logger=logger,
        )

    def _create(self,) -> Model:
        if self.model_type == "chaii-xlmrb-1":
            model = ChaiiXLMRBModel1(
                pretrained_model_name_or_path=self.pretrained_model_name_or_path,
                warmup_epoch=self.warmup_epoch,
                logger=self.logger,
            )
        else:
            raise NotImplementedError(f"model_type {self.model_type} is not supported.")
        return model
