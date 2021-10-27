from transformers.models.auto.tokenization_auto import AutoTokenizer

from src.factory import Factory
from src.log import myLogger
from src.preprocessor.preprocessor import (BaselineKernelPreprocessorV1,
                                           BaselineKernelPreprocessorV2,
                                           BaselineKernelPreprocessorV3,
                                           BaselineKernelPreprocessorV4,
                                           BaselineKernelPreprocessorV5,
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
        use_language_as_question: bool,
        debug: bool,
        logger: myLogger,
    ) -> None:
        super().__init__(
            preprocessor_type=preprocessor_type,
            tokenizer_type=tokenizer_type,
            max_length=max_length,
            pad_on_right=pad_on_right,
            stride=stride,
            use_language_as_question=use_language_as_question,
            debug=debug,
            logger=logger,
        )

    def _create(self, data_repository: DataRepository) -> Preprocessor:
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_type)

        if self.preprocessor_type == "baseline_kernel_v1":
            preprocessor = BaselineKernelPreprocessorV1(
                tokenizer=tokenizer,
                data_repository=data_repository,
                max_length=self.max_length,
                pad_on_right=self.pad_on_right,
                stride=self.stride,
                use_language_as_question=self.use_language_as_question,
                debug=self.debug,
                logger=self.logger,
            )
        elif self.preprocessor_type == "baseline_kernel_v2":
            preprocessor = BaselineKernelPreprocessorV2(
                tokenizer=tokenizer,
                data_repository=data_repository,
                max_length=self.max_length,
                pad_on_right=self.pad_on_right,
                stride=self.stride,
                use_language_as_question=self.use_language_as_question,
                debug=self.debug,
                logger=self.logger,
            )
        elif self.preprocessor_type == "baseline_kernel_v3":
            preprocessor = BaselineKernelPreprocessorV3(
                tokenizer=tokenizer,
                data_repository=data_repository,
                max_length=self.max_length,
                pad_on_right=self.pad_on_right,
                stride=self.stride,
                use_language_as_question=self.use_language_as_question,
                debug=self.debug,
                logger=self.logger,
            )
        elif self.preprocessor_type == "baseline_kernel_v4":
            preprocessor = BaselineKernelPreprocessorV4(
                tokenizer=tokenizer,
                data_repository=data_repository,
                max_length=self.max_length,
                pad_on_right=self.pad_on_right,
                stride=self.stride,
                use_language_as_question=self.use_language_as_question,
                debug=self.debug,
                logger=self.logger,
            )
        elif self.preprocessor_type == "baseline_kernel_v5":
            preprocessor = BaselineKernelPreprocessorV5(
                tokenizer=tokenizer,
                data_repository=data_repository,
                max_length=self.max_length,
                pad_on_right=self.pad_on_right,
                stride=self.stride,
                use_language_as_question=self.use_language_as_question,
                debug=self.debug,
                logger=self.logger,
            )
        else:
            raise NotImplementedError(
                f"preprocessor_type {self.preprocessor_type} is not supported."
            )
        return preprocessor
