from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.factory import Factory
from src.log import myLogger
from src.preprocessor.preprocessor import (BaselineKernelPreprocessor,
                                           Preprocessor)
from src.repository.data_repository import DataRepository


class PreprocessorFactory(Factory[Preprocessor]):
    def __init__(
        self,
        preprocessor_type: str,
        tokenizer_type: str,
        max_length: int,
        pad_on_right: bool,
        stride: int,
        debug: bool,
        logger: myLogger,
    ) -> None:
        super().__init__(
            preprocessor_type=preprocessor_type,
            tokenizer_type=tokenizer_type,
            max_length=max_length,
            pad_on_right=pad_on_right,
            stride=stride,
            debug=debug,
            logger=logger,
        )

    def _create(self, data_repository: DataRepository) -> Preprocessor:
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_type)

        if self.preprocessor_type.startswith("baseline_kernel"):
            preprocessor = BaselineKernelPreprocessor(
                tokenizer=tokenizer,
                data_repository=data_repository,
                max_length=self.max_length,
                pad_on_right=self.pad_on_right,
                stride=self.stride,
                debug=self.debug,
                logger=self.logger,
            )
        else:
            raise NotImplementedError(
                f"preprocessor_type {self.preprocessor_type} is not supported."
            )
        return preprocessor
