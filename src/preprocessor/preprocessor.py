import os
from abc import ABCMeta, abstractmethod
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple

import pandas as pd
from pandas import DataFrame, Series
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from src.data.repository import DataRepository
from src.log import myLogger
from src.timer import class_dec_timer


class Preprocessor(metaclass=ABCMeta):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_repository: DataRepository,
        max_length: int,
        is_test: bool,
        logger: myLogger,
    ) -> None:
        self.tokenizer = tokenizer
        self.data_repository = data_repository
        self.max_length = max_length
        self.is_test = is_test
        self.logger = logger

    @abstractmethod
    def __call__(self, df: DataFrame) -> DataFrame:
        raise NotImplementedError()


class PreprocessorV1(Preprocessor):
    @class_dec_timer(unit="m")
    def __call__(self, df: DataFrame) -> DataFrame:
        self.logger.info("now preprocessing df ...")
        # reset index to deal it correctly
        df.reset_index(drop=True, inplace=True)
        with Pool(os.cpu_count()) as p:
            iter_func = partial(
                _prep_text_v1,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                is_test=self.is_test,
            )
            imap = p.imap_unordered(iter_func, df.iterrows())
            res_pairs = list(tqdm(imap, total=len(df)))
            p.close()
            p.join()
        res_df = pd.concat([row for _, row in sorted(res_pairs)])
        self.data_repository.save_preprocessed_df(res_df, ver="v1")
        self.logger.info("done.")
        return res_df


# define outside of the class becuase of Pool restriction
def _prep_text_v1(
    row_pair: Tuple[int, Series],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    is_test: bool,
) -> Tuple[int, Series]:
    i, row = row_pair
    tokenized_res = tokenizer.encode_plus(
        text=str(row["context"]),
        text_pair=str(row["question"]),
        add_special_tokens=True,
        padding="max_length",
        truncation="only_first",
        max_length=max_length,
        stride=0,
        return_tensor="pt",
        return_attention_mask=True,
        return_overflowing_tokens=True,
        return_special_tokens_mask=False,
        return_offsets_mapping=True,
        return_length=True,
    )
    input_ids: List[int] = tokenized_res["input_ids"]
    attention_mask: List[int] = tokenized_res["attention_mask"]
    special_tokens_mask: List[int] = tokenized_res["special_tokens_mask"]
    sequence_ids: List[int] = tokenized_res["sequence_ids"]
    offset_mapping: List[Tuple[int, int]] = tokenized_res["offset_mapping"]
    cls_index = input_ids.index(tokenizer.cls_token_id)
    row["input_ids"] = input_ids
    row["attention_mask"] = attention_mask
    row["special_tokens_mask"] = special_tokens_mask
    row["sequence_ids"] = sequence_ids
    row["offset_mapping"] = offset_mapping
    if is_test:
        row["start_position"] = cls_index
        row["end_position"] = cls_index
    else:
        start_char_index = int(row["answer_start"])
        end_char_index = start_char_index + len(row["answer_text"])

        # Start token index of the current span in the text.
        token_start_index = 0
        while sequence_ids[token_start_index] != 0:
            token_start_index += 1

        # End token index of the current span in the text.
        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 0:
            token_end_index -= 1

        # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
        if not (
            offset_mapping[token_start_index][0] <= start_char_index
            and offset_mapping[token_end_index][1] >= end_char_index
        ):
            row["start_position"] = cls_index
            row["end_position"] = cls_index
        else:
            # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
            # Note: we could go after the last offset if the answer is the last word (edge case).
            while (
                token_start_index < len(offset_mapping)
                and offset_mapping[token_start_index][0] <= start_char_index
            ):
                token_start_index += 1
            row["start_positions"] = token_start_index - 1  # -1 because +1 even in == case
            while offset_mapping[token_end_index][1] >= end_char_index:
                token_end_index -= 1
            row["end_positions"] = token_end_index + 1  # +1 because even in == case
    return i, row
