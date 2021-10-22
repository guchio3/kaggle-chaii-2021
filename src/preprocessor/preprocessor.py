import itertools
import os
from abc import ABCMeta, abstractmethod
# from functools import partial
# from multiprocessing import Pool
from typing import List, Tuple

import pandas as pd
from pandas import DataFrame, Series
from tqdm.auto import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer

from src.log import myLogger
from src.repository.data_repository import DataRepository
from src.timer import class_dec_timer


class Preprocessor(metaclass=ABCMeta):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_repository: DataRepository,
        max_length: int,
        is_test: bool,
        debug: bool,
        logger: myLogger,
    ) -> None:
        self.tokenizer = tokenizer
        self.data_repository = data_repository
        self.max_length = max_length
        self.is_test = is_test
        self.debug = debug
        self.logger = logger

    @abstractmethod
    def __call__(self, df: DataFrame) -> DataFrame:
        raise NotImplementedError()


class BaselineKernelPreprocessor(Preprocessor):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_repository: DataRepository,
        max_length: int,
        is_test: bool,
        pad_on_right: bool,
        stride: int,
        debug: bool,
        logger: myLogger,
    ):
        super().__init__(
            tokenizer=tokenizer,
            data_repository=data_repository,
            max_length=max_length,
            is_test=is_test,
            debug=debug,
            logger=logger,
        )
        self.pad_on_right = pad_on_right
        self.stride = stride
        self.ver = self.build_ver(
            max_length=max_length, pad_on_right=pad_on_right, stride=stride
        )

    def build_ver(self, max_length: int, pad_on_right: bool, stride: int) -> str:
        ver = f"BaselineKernel_max_length_{max_length}_pad_on_right_{pad_on_right}_stride_{stride}"
        self.logger.info(f"ver => {ver}")
        return ver

    @class_dec_timer(unit="m")
    def __call__(self, df: DataFrame) -> DataFrame:
        if not self.is_test and self.data_repository.preprocessed_df_exists(
            ver=self.ver
        ):
            self.logger.info("load preprocessed_df because it already exists.")
            res_df = self.data_repository.load_preprocessed_df(ver=self.ver)
        else:
            self.logger.info("now preprocessing df ...")
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            # reset index to deal it correctly
            df.reset_index(drop=True, inplace=True)
            if self.debug:
                res_lists = []
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    res_lists.append(
                        self._prep_text_v1(
                            row_pair=(i, row),
                            tokenizer=self.tokenizer,
                            max_length=self.max_length,
                            pad_on_right=self.pad_on_right,
                            stride=self.stride,
                            is_test=self.is_test,
                        )
                    )
            else:
                res_lists = []
                for i, row in tqdm(df.iterrows(), total=len(df)):
                    res_lists.append(
                        self._prep_text_v1(
                            row_pair=(i, row),
                            tokenizer=self.tokenizer,
                            max_length=self.max_length,
                            pad_on_right=self.pad_on_right,
                            stride=self.stride,
                            is_test=self.is_test,
                        )
                    )
                # with Pool(os.cpu_count()) as p:
                #     iter_func = partial(
                #         _prep_text_v1,
                #         tokenizer=self.tokenizer,
                #         max_length=self.max_length,
                #         is_test=self.is_test,
                #     )
                #     imap = p.imap_unordered(iter_func, df.iterrows())
                #     res_pairs = list(tqdm(imap, total=len(df)))
                #     p.close()
                #     p.join()
            sorted_res_pairs = sorted(list(itertools.chain.from_iterable(res_lists)))
            res_df = pd.DataFrame(
                [row.to_dict() for _, _, row, _ in sorted(sorted_res_pairs)]
            )
            successed_cnt = len(
                list(filter(lambda res_pair: res_pair[3], sorted_res_pairs))
            )
            self.logger.info(
                f"successed_ratio: {successed_cnt} / {len(sorted_res_pairs)}"
            )
            if not self.debug and not self.is_test:
                self.data_repository.save_preprocessed_df(res_df, ver=self.ver)
            else:
                self.logger.info(
                    "ignore save_preprocessed_df because either of "
                    f"self.debug {self.debug} self.is_test {self.is_test}."
                )
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self.logger.info("done.")
        return res_df

    def _prep_text_v1(
        self,
        row_pair: Tuple[int, Series],
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        pad_on_right: bool,
        stride: int,
        is_test: bool,
    ) -> List[Tuple[int, int, Series, bool]]:
        is_successed = True
        if tokenizer.cls_token_id is not None:
            cls_token_id = tokenizer.cls_token_id
        else:
            raise Exception("no cls_token_id.")
        context_index = 1 if pad_on_right else 0

        i, row = row_pair
        row["question"] = str(row["question"]).lstrip()
        tokenized_res = tokenizer.encode_plus(
            text=str(row["question" if pad_on_right else "context"]),
            text_pair=str(row["context" if pad_on_right else "question"]),
            padding="max_length",
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=stride,
            return_attention_mask=True,
            return_overflowing_tokens=True,
            return_special_tokens_mask=False,
            return_offsets_mapping=True,
        )

        reses = []
        for j, (input_ids, attention_mask, offset_mapping) in enumerate(
            zip(
                tokenized_res["input_ids"],
                tokenized_res["attention_mask"],
                tokenized_res["offset_mapping"],
            )
        ):
            # special_tokens_mask: List[int] = tokenized_res["special_tokens_mask"]
            # sequence_ids: List[int] = tokenized_res["sequence_ids"]
            token_type_ids: List[int] = tokenized_res.sequence_ids(j)  # CAUTION!!!!!
            sequence_ids = [i for i in range(len(input_ids))]

            cls_index = input_ids.index(cls_token_id)
            row["input_ids"] = input_ids
            row["token_type_ids"] = token_type_ids
            row["attention_mask"] = attention_mask
            # row["special_tokens_mask"] = special_tokens_mask
            row["sequence_ids"] = sequence_ids
            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            row["offset_mapping"] = [
                (o if token_type_ids[k] == context_index else (-1, -1))
                for k, o in enumerate(offset_mapping)
            ]
            if is_test:
                is_successed = False
                row["start_position"] = cls_index
                row["end_position"] = cls_index
                row["segmentation_position"] = [1] + [0] * (len(offset_mapping) - 1)
            else:
                start_char_index = int(row["answer_start"])
                end_char_index = start_char_index + len(row["answer_text"])

                # Start token index of the current span in the text.
                token_start_index = 0
                while token_type_ids[token_start_index] != 0:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while token_type_ids[token_end_index] != 0:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (
                    offset_mapping[token_start_index][0] <= start_char_index
                    and offset_mapping[token_end_index][1] >= end_char_index
                ):
                    is_successed = False
                    row["start_position"] = cls_index
                    row["end_position"] = cls_index
                    row["segmentation_position"] = [1] + [0] * (len(offset_mapping) - 1)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while (
                        token_start_index < len(offset_mapping)
                        and offset_mapping[token_start_index][0] <= start_char_index
                    ):
                        token_start_index += 1
                    row["start_position"] = (
                        token_start_index - 1
                    )  # -1 because +1 even in == case
                    while offset_mapping[token_end_index][1] >= end_char_index:
                        token_end_index -= 1
                    row["end_position"] = (
                        token_end_index + 1
                    )  # +1 because even in == case
                    row["segmentation_position"] = [
                        1
                        if row["start_position"] <= i and i <= row["end_position"]
                        else 0
                        for i in range(len(offset_mapping))
                    ]
            reses.append((i, j, row, is_successed))
        return reses
