from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar

T = TypeVar("T")


class Factory(Generic[T], metaclass=ABCMeta):
    def __init__(self, **kwargs):
        self.__set_or_update_attrs(**kwargs)

    def create(self, order_settings: Optional[Dict[str, Any]] = None, **kwargs) -> T:
        if order_settings:
            self.__set_or_update_attrs(**order_settings)
        return self._create(**kwargs)

    @abstractmethod
    def _create(self, **kwargs) -> T:
        raise NotImplementedError()

    def __set_or_update_attrs(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
