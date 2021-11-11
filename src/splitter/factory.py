from src.factory import Factory
from src.log import myLogger
from src.splitter.splitter import (GKFSplitter, KFSplitter, SKFSplitter,
                                   Splitter)


class SplitterFactory(Factory):
    def __init__(
        self,
        splitter_type: str,
        split_num: int,
        shuffle: bool,
        random_state: int,
        logger: myLogger,
    ) -> None:
        self.splitter_type = splitter_type
        self.split_num = split_num
        self.shuffle = shuffle
        self.random_state = random_state
        self.logger = logger

    def _create(self,) -> Splitter:
        if self.splitter_type == "kf":
            return KFSplitter(
                split_num=self.split_num,
                shuffle=self.shuffle,
                random_state=self.random_state,
                logger=self.logger,
            )
        # elif split_type == "skf":
        #     return SKFSplitter(
        #         split_num=split_num,
        #         shuffle=shuffle,
        #         random_state=random_state,
        #         logger=self.logger,
        #     )
        elif self.splitter_type == "gkf":
            return GKFSplitter(
                split_num=self.split_num,
                shuffle=self.shuffle,
                random_state=self.random_state,
                logger=self.logger,
            )
        else:
            raise NotImplementedError(
                f"splitter_type `{self.splitter_type}` is not implemented"
            )
