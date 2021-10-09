from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional, Tuple

from sklearn.model_selection import GroupKFold as gkf
from sklearn.model_selection import KFold as kf
from sklearn.model_selection import StratifiedKFold as skf

from src.log import myLogger


class Splitter(metaclass=ABCMeta):
    def __init__(
        self, split_num: int, shuffle: bool, random_state: int, logger: myLogger
    ):
        self.logger = logger
        self.split_num = split_num
        self.shuffle = shuffle
        self.random_state = random_state

    @abstractmethod
    def split(
        self, x: List[List[Any]], y: List[Any], groups: Optional[List[Any]]
    ) -> Tuple[List[int], List[int]]:
        raise NotImplementedError()


class KFSplitter(Splitter):
    def split(
        self, x: List[List[Any]], y: List[Any], groups: Optional[List[Any]] = None
    ) -> Tuple[List[int], List[int]]:
        trn_val_idxes: Tuple[List[int], List[int]] = kf(
            self.split_num, shuffle=self.shuffle, random_state=self.random_state
        ).split(x, y)
        return trn_val_idxes


class SKFSplitter(Splitter):
    def split(
        self, x: List[List[Any]], y: List[Any], groups: Optional[List[Any]] = None
    ) -> Tuple[List[int], List[int]]:
        trn_val_idxes: Tuple[List[int], List[int]] = skf(
            self.split_num, shuffle=self.shuffle, random_state=self.random_state
        ).split(x, y)
        return trn_val_idxes


class GKFSplitter(Splitter):
    def split(
        self, x: List[List[Any]], y: List[Any], groups: Optional[List[Any]] = None
    ) -> Tuple[List[int], List[int]]:
        trn_val_idxes: Tuple[List[int], List[int]] = gkf(self.split_num).split(
            x, y, groups
        )
        return trn_val_idxes
