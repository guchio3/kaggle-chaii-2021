from src.factory import Factory
from src.log import myLogger
from src.model import Model


class ModelFactory(Factory[Model]):
    def __init__(
        self,
        model_type: str,
        img_size: int,
        img_channel: int,
        do_rate: float,
        weights: str,
        logger: myLogger,
        **kwargs
    ):
        super().__init__(
            model_type=model_type,
            img_size=img_size,
            img_channel=img_channel,
            do_rate=do_rate,
            weights=weights,
            logger=logger,
            **kwargs
        )

    def _create(
        self,
    ) -> Model:
        if self.model_type == "chaii-xlmrb-1":
            model = ChaiiXLMRBModel1()
        else:
            raise NotImplementedError(f"model_type {self.model_type} is not supported.")
        return model
