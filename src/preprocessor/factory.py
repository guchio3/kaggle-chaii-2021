from transformers import AutoTokenizer

from src.factory import Factory
from src.log import myLogger
from src.preprocessor.preprocessor import Preprocessor, PreprocessorV1
from src.repository.data_repository import DataRepository


class PreprocessorFactory(Factory[Preprocessor]):
    def __init__(
        self,
        preprocessor_type: str,
        tokenizer_type: str,
        max_length: int,
        is_test: bool,
        logger: myLogger,
    ) -> None:
        super().__init__(
            preprocessor_type=preprocessor_type,
            tokenizer_type=tokenizer_type,
            max_length=max_length,
            is_test=is_test,
            logger=logger,
        )

    def _create(self, data_repository: DataRepository) -> Preprocessor:
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_type)

        if self.preprocessor_type == "v1":
            preprocessor = PreprocessorV1(
                tokenizer=tokenizer,
                data_repository=data_repository,
                max_length=self.max_length,
                is_test=self.is_test,
                logger=self.logger,
            )
        else:
            raise NotImplementedError(
                f"preprocessor_type {self.preprocessor_type} is not supported."
            )
        return preprocessor
