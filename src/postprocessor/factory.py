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
        use_chars_length: bool,
        text_postprocess: Optional[str],
        use_multiprocess: bool,
        logger: myLogger,
    ) -> None:
        super().__init__(
            postprocessor_type=postprocessor_type,
            logger=logger,
            n_best_size=n_best_size,
            max_answer_length=max_answer_length,
            use_chars_length=use_chars_length,
            text_postprocess=text_postprocess,
            use_multiprocess=use_multiprocess,
        )

    def _create(self) -> Postprocessor:
        if self.postprocessor_type == "baseline_kernel":
            preprocessor = BaselineKernelPostprocessor(
                n_best_size=self.n_best_size,
                max_answer_length=self.max_answer_length,
                use_chars_length=self.use_chars_length,
                text_postprocess=self.text_postprocess,
                use_multiprocess=self.use_multiprocess,
                logger=self.logger,
            )
        else:
            raise NotImplementedError(
                f"postprocessor_type {self.postprocessor_type} is not supported."
            )
        return preprocessor
