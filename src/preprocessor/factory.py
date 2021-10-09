from transformers import PreTrainedTokenizer

from src.data.repository import DataRepository
from src.factory import Factory
from src.log import myLogger
from src.preprocessor.preprocessor import Preprocessor, PreprocessorV1


class PreprocessorFactory(Factory):
    def __init__(
        self, preprocessor_type: str, max_length: int, is_test: bool, logger: myLogger
    ) -> None:
        self.preprocessor_type = preprocessor_type
        self.max_length = max_length
        self.is_test = is_test
        self.logger = logger

    def create(
        self, tokenizer: PreTrainedTokenizer, data_repository: DataRepository
    ) -> Preprocessor:
        if self.preprocessor_type == "v1":
            preprocessor = PreprocessorV1(
                tokenizer=tokenizer,
                data_repository=data_repository,
                max_length=self.max_length,
                is_test=self.is_test,
                logger=self.logger,
            )
        else:
            raise NotImplementedError()
        return preprocessor
