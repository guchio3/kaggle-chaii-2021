from src.factory import Factory
from src.log import myLogger
from src.postprocessor.postprocessor import (BaselineKernelPostprocessor,
                                             Postprocessor)


class PostprocessorFactory(Factory[Postprocessor]):
    def __init__(
        self,
        postprocessor_type: str,
        logger: myLogger,
    ) -> None:
        super().__init__(
            postprocessor_type=postprocessor_type,
            logger=logger,
        )

    def _create(self) -> Postprocessor:
        if self.postprocessor_type == "baseline_kernel":
            preprocessor = BaselineKernelPostprocessor(
                logger=self.logger,
            )
        else:
            raise NotImplementedError(
                f"postprocessor_type {self.postprocessor_type} is not supported."
            )
        return preprocessor
