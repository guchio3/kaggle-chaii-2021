from abc import ABCMeta, abstractmethod

from torch.nn import Module
from transformers import AutoModel


class Model(Module, metaclass=ABCMeta):
    def __init__(self, pretrained_model_name_or_path: str) -> None:
        super().__init__()
        if isinstance(pretrained_model_name_or_path, str):
            self.model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        else:
            # for sub
            self.model = AutoModel(pretrained_model_name_or_path)

    @abstractmethod
    def calc_loss() -> float:
        raise NotImplementedError()

    def resize_token_embeddings(self, token_num: int) -> None:
        self.model.resize_token_embeddings(token_num)
