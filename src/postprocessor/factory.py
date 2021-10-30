from typing import Optional

from src.factory import Factory
from src.log import myLogger
from src.postprocessor.postprocessor import (BaselineKernelPostprocessor,
                                             Postprocessor)


class PostprocessorFactory(Factory[Postprocessor]):
    def __init__(
        self,
        postprocessor_type: str,
        n_best_size: int,
        max_answer_length: int,
        text_postprocess: Optional[str],
        logger: myLogger,
    ) -> None:
        super().__init__(
            postprocessor_type=postprocessor_type,
            logger=logger,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
            text_postprocess=text_postprocess,
        )

    def _create(self) -> Postprocessor:
        if self.postprocessor_type == "baseline_kernel":
            preprocessor = BaselineKernelPostprocessor(
                n_best_size=self.n_best_size,
                max_answer_length=self.max_answer_length,
                text_postprocess=self.text_postprocess,
                logger=self.logger,
            )
        else:
            raise NotImplementedError(
                f"postprocessor_type {self.postprocessor_type} is not supported."
            )
        return preprocessor
