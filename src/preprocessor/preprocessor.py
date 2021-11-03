import itertools
import os
import re
from abc import ABCMeta, abstractmethod
from copy import deepcopy
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
        debug: bool,
        logger: myLogger,
    ) -> None:
        self.tokenizer = tokenizer
        self.data_repository = data_repository
        self.max_length = max_length
        self.debug = debug
        self.logger = logger

    @abstractmethod
    def __call__(
        self, df: DataFrame, dataset_name: str, enforce_preprocess: bool, is_test: bool
    ) -> DataFrame:
        raise NotImplementedError()


class BaselineKernelPreprocessor(Preprocessor, metaclass=ABCMeta):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_repository: DataRepository,
        max_length: int,
        pad_on_right: bool,
        stride: int,
        use_language_as_question: bool,
        add_overflowing_batch_id: bool,
        debug: bool,
        logger: myLogger,
    ):
        super().__init__(
            tokenizer=tokenizer,
            data_repository=data_repository,
            max_length=max_length,
            debug=debug,
            logger=logger,
        )
        self.pad_on_right = pad_on_right
        self.stride = stride
        self.use_language_as_question = use_language_as_question
        self.add_overflowing_batch_id = add_overflowing_batch_id

    @class_dec_timer(unit="m")
    def __call__(
        self, df: DataFrame, dataset_name: str, enforce_preprocess: bool, is_test: bool
    ) -> DataFrame:
        if (
            not enforce_preprocess
            and not is_test
            and self.data_repository.preprocessed_df_exists(
                dataset_name=dataset_name,
                class_name=self.__class__.__name__,
                tokenizer_name=self.tokenizer.__class__.__name__,
                max_length=self.max_length,
                pad_on_right=self.pad_on_right,
                stride=self.stride,
                use_language_as_question=self.use_language_as_question,
                add_overflowing_batch_id=self.add_overflowing_batch_id,
            )
        ):
            self.logger.info("load preprocessed_df because it already exists.")
            res_df = self.data_repository.load_preprocessed_df(
                dataset_name=dataset_name,
                class_name=self.__class__.__name__,
                tokenizer_name=self.tokenizer.__class__.__name__,
                max_length=self.max_length,
                pad_on_right=self.pad_on_right,
                stride=self.stride,
                use_language_as_question=self.use_language_as_question,
                add_overflowing_batch_id=self.add_overflowing_batch_id,
            )
        else:
            self.logger.info("now preprocessing df ...")
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            # reset index to deal it correctly
            df.reset_index(drop=True, inplace=True)
            res_lists = []
            for i, row in tqdm(df.iterrows(), total=len(df)):
                res_lists.append(
                    self._prep_text_v1(
                        row_pair=(i, row),
                        tokenizer=self.tokenizer,
                        max_length=self.max_length,
                        pad_on_right=self.pad_on_right,
                        stride=self.stride,
                        use_language_as_question=self.use_language_as_question,
                        add_overflowing_batch_id=self.add_overflowing_batch_id,
                        is_test=is_test,
                    )
                )
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
            if not self.debug and not is_test:
                self.data_repository.save_preprocessed_df(
                    dataset_name=dataset_name,
                    preprocessed_df=res_df,
                    class_name=self.__class__.__name__,
                    tokenizer_name=self.tokenizer.__class__.__name__,
                    max_length=self.max_length,
                    pad_on_right=self.pad_on_right,
                    stride=self.stride,
                    use_language_as_question=self.use_language_as_question,
                    add_overflowing_batch_id=self.add_overflowing_batch_id,
                )
            else:
                self.logger.info(
                    "ignore save_preprocessed_df because either of "
                    f"self.debug {self.debug} is_test {is_test}."
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
        use_language_as_question: bool,
        add_overflowing_batch_id: bool,
        is_test: bool,
    ) -> List[Tuple[int, int, Series, bool]]:
        if tokenizer.cls_token_id is not None:
            cls_token_id = tokenizer.cls_token_id
        else:
            raise Exception("no cls_token_id.")
        context_index = 1 if pad_on_right else 0

        i, row = row_pair
        context = str(row["context"])
        question = self._prep_question(
            row=row, use_language_as_question=use_language_as_question
        )
        tokenized_res = tokenizer.encode_plus(
            text=question if pad_on_right else context,
            text_pair=context if pad_on_right else question,
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
            if add_overflowing_batch_id:
                min_s = len(context) * 5
                max_e = -1
                temp_token_type_ids: List[int] = tokenized_res.sequence_ids(
                    j
                )  # CAUTION!!!!!
                for temp_token_type_id, (temp_s, temp_e) in zip(
                    temp_token_type_ids, offset_mapping
                ):
                    if temp_token_type_id == context_index:
                        min_s = min(temp_s, min_s)
                        max_e = max(temp_e, max_e)
                context_j = context[min_s:max_e]
                tokenized_res_j = tokenizer.encode_plus(
                    text=f"{j} {question}" if pad_on_right else context_j,
                    text_pair=context_j if pad_on_right else f"{j} {question}",
                    padding="max_length",
                    truncation="only_second" if pad_on_right else "only_first",
                    max_length=max_length + 5,
                    stride=stride,
                    return_attention_mask=True,
                    return_overflowing_tokens=True,
                    return_special_tokens_mask=False,
                    return_offsets_mapping=True,
                )
                input_ids = tokenized_res_j["input_ids"][0]
                attention_mask = tokenized_res_j["attention_mask"][0]
                offset_mapping = tokenized_res_j["offset_mapping"][0]
                token_type_ids: List[int] = tokenized_res_j.sequence_ids(
                    0
                )  # CAUTION!!!!!
            else:
                token_type_ids: List[int] = tokenized_res.sequence_ids(
                    j
                )  # CAUTION!!!!!

            is_successed = True
            row_j = deepcopy(row)
            # token_type_ids: List[int] = tokenized_res.sequence_ids(j)  # CAUTION!!!!!
            sequence_ids = [i for i in range(len(input_ids))]

            cls_index = input_ids.index(cls_token_id)
            row_j["input_ids"] = input_ids
            row_j["token_type_ids"] = token_type_ids
            row_j["attention_mask"] = attention_mask
            row_j["sequence_ids"] = sequence_ids
            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            row_j["offset_mapping"] = [
                (o if token_type_ids[k] == context_index else (-1, -1))
                for k, o in enumerate(offset_mapping)
            ]
            row_j["overflowing_batch_id"] = j  # jth batch
            if is_test:
                is_successed = False
                row_j["answer_text"] = ""
                row_j["start_position"] = cls_index
                row_j["end_position"] = cls_index
                row_j["segmentation_position"] = [1] + [0] * (len(offset_mapping) - 1)
                row_j["is_contain_answer_text"] = 0
            else:
                start_char_index = self._start_char_index(row=row_j)
                end_char_index = start_char_index + len(row_j["answer_text"])

                # Start token index of the current span in the text.
                token_start_index = 0
                while token_type_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while token_type_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (
                    offset_mapping[token_start_index][0] <= start_char_index
                    and offset_mapping[token_end_index][1] >= end_char_index
                ):
                    is_successed = False
                    row_j["start_position"] = cls_index
                    row_j["end_position"] = cls_index
                    row_j["segmentation_position"] = [1] + [0] * (
                        len(offset_mapping) - 1
                    )
                    row_j["is_contain_answer_text"] = 0
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while (
                        token_start_index < len(offset_mapping)
                        and offset_mapping[token_start_index][0] <= start_char_index
                    ):
                        token_start_index += 1
                    row_j["start_position"] = (
                        token_start_index - 1
                    )  # -1 because +1 even in == case
                    while offset_mapping[token_end_index][1] >= end_char_index:
                        token_end_index -= 1
                    row_j["end_position"] = (
                        token_end_index + 1
                    )  # +1 because even in == case
                    row_j["segmentation_position"] = [
                        1
                        if row_j["start_position"] <= i and i <= row_j["end_position"]
                        else 0
                        for i in range(len(offset_mapping))
                    ]
                    row_j["is_contain_answer_text"] = 1
            reses.append((i, j, row_j, is_successed))
        reses = self._pre_postprocess(preprocessed_results=reses)
        return reses

    def _prep_question(
        self,
        row: Series,
        # tokenizer: PreTrainedTokenizer,
        use_language_as_question: bool,
    ) -> str:
        question = str(row["question"]).lstrip()
        if use_language_as_question:
            # tokenizer.add_tokens(["<l>", "</l>"])
            language = str(row["language"]).lstrip()
            question = f"{language} </s> {question}"
        return question

    @abstractmethod
    def _start_char_index(self, row: Series) -> int:
        raise NotImplementedError()

    def _pre_postprocess(
        self, preprocessed_results: List[Tuple[int, int, Series, bool]]
    ) -> List[Tuple[int, int, Series, bool]]:
        return preprocessed_results


class BaselineKernelPreprocessorV1(BaselineKernelPreprocessor):
    def _start_char_index(self, row: Series) -> int:
        start_char_index = int(row["answer_start"])
        return start_char_index


class BaselineKernelPreprocessorV2(BaselineKernelPreprocessor):
    def _start_char_index(self, row: Series) -> int:
        context = str(row["context"])

        context_start_char_index = len(context)
        context_end_char_index = 0
        for offset_s, offset_e in row["offset_mapping"]:
            if offset_s == -1:
                continue
            context_start_char_index = min(offset_s, context_start_char_index)
            context_end_char_index = max(offset_e, context_end_char_index)
        if context_start_char_index == len(context) or context_end_char_index == 0:
            raise Exception("failed to search context_part.")
        context_part = context[context_start_char_index:context_end_char_index]

        answer_text = str(row["answer_text"])
        try:
            search_res = re.search(answer_text, context_part)
        except Exception as e:
            self.logger.warn(e)
            search_res = None
        # no match or exception case
        if search_res is None:
            start_char_index = int(row["answer_start"])
        else:
            start_char_index = context_start_char_index + int(search_res.span()[0])
        return start_char_index


class BaselineKernelPreprocessorV3(BaselineKernelPreprocessorV2):
    def _pre_postprocess(
        self, preprocessed_results: List[Tuple[int, int, Series, bool]]
    ) -> List[Tuple[int, int, Series, bool]]:
        cls_index = 0
        res_preprocessed_results = []
        id_succeeded_cnt = 0
        for i, j, row, is_successed in preprocessed_results:
            if is_successed:
                id_succeeded_cnt += 1
            if id_succeeded_cnt > 10:
                is_successed = False
                row["start_position"] = cls_index
                row["end_position"] = cls_index
                row["segmentation_position"] = [1] + [0] * (
                    len(row["offset_mapping"]) - 1
                )
            res_preprocessed_results.append((i, j, row, is_successed))
        return preprocessed_results


class BaselineKernelPreprocessorV4(BaselineKernelPreprocessorV2):
    def _pre_postprocess(
        self, preprocessed_results: List[Tuple[int, int, Series, bool]]
    ) -> List[Tuple[int, int, Series, bool]]:
        cls_index = 0
        res_preprocessed_results = []
        id_succeeded_cnt = 0
        for i, j, row, is_successed in preprocessed_results:
            if is_successed:
                id_succeeded_cnt += 1
            if id_succeeded_cnt > 2:
                is_successed = False
                row["start_position"] = cls_index
                row["end_position"] = cls_index
                row["segmentation_position"] = [1] + [0] * (
                    len(row["offset_mapping"]) - 1
                )
            res_preprocessed_results.append((i, j, row, is_successed))
        return preprocessed_results


class BaselineKernelPreprocessorV5(BaselineKernelPreprocessorV2):
    def _pre_postprocess(
        self, preprocessed_results: List[Tuple[int, int, Series, bool]]
    ) -> List[Tuple[int, int, Series, bool]]:
        cls_index = 0
        res_preprocessed_results = []
        id_succeeded_cnt = 0
        for i, j, row, is_successed in preprocessed_results:
            if is_successed:
                id_succeeded_cnt += 1
            if id_succeeded_cnt > 1:
                is_successed = False
                row["start_position"] = cls_index
                row["end_position"] = cls_index
                row["segmentation_position"] = [1] + [0] * (
                    len(row["offset_mapping"]) - 1
                )
            res_preprocessed_results.append((i, j, row, is_successed))
        return preprocessed_results
