import os
import re
from abc import ABCMeta, abstractmethod
from functools import partial
from multiprocessing import Pool
from string import punctuation
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from torch import Tensor
from tqdm.auto import tqdm

from src.log import myLogger


class Postprocessor(metaclass=ABCMeta):
    def __init__(
        self,
        n_best_size: int,
        max_answer_length: int,
        use_chars_length: bool,
        text_postprocess: Optional[str],
        use_multiprocess: bool,
        logger: myLogger,
    ) -> None:
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.use_chars_length = use_chars_length
        self.text_postprocess = text_postprocess
        self.use_multiprocess = use_multiprocess
        self.logger = logger

    @abstractmethod
    def __call__(
        self,
        ids: List[str],
        contexts: List[str],
        answer_texts: List[str],
        offset_mappings: List[List[Tuple[int, int]]],
        start_logits: List[Tensor],
        end_logits: List[Tensor],
    ) -> Tuple[List[str], List[str], List[str]]:
        raise NotImplementedError()


class BaselineKernelPostprocessor(Postprocessor):
    def __call__(
        self,
        ids: List[str],
        contexts: List[str],
        answer_texts: List[str],
        offset_mappings: List[List[Tuple[int, int]]],
        start_logits: List[Tensor],
        end_logits: List[Tensor],
    ) -> Tuple[List[str], List[str], List[str]]:
        self.logger.info("start postprocessing")

        if (
            len(ids) != len(contexts)
            or len(contexts) != len(answer_texts)
            or len(answer_texts) != len(offset_mappings)
            or len(offset_mappings) != len(start_logits)
            or len(start_logits) != len(end_logits)
        ):
            raise Exception(
                "len of ids, contexts, answer_texts, offset_mappings, start_logits, and end_logits are different. "
                f"len(ids): {len(ids)}, "
                f"len(contexts): {len(contexts)}, "
                f"len(answer_texts): {len(answer_texts)}, "
                f"len(offset_mappings): {len(offset_mappings)}, "
                f"len(start_logits): {len(start_logits)}, "
                f"len(end_logits): {len(end_logits)}."
            )

        raw_df = DataFrame()
        raw_df["id"] = ids
        raw_df["context"] = contexts
        raw_df["answer_text"] = answer_texts
        raw_df["offset_mapping"] = offset_mappings
        raw_df["start_logit"] = start_logits
        raw_df["end_logit"] = end_logits

        if self.use_multiprocess:
            with Pool(os.cpu_count()) as p:
                iter_func = partial(
                    _apply_extract_best_answer_pred,
                    n_best_size=self.n_best_size,
                    max_answer_length=self.max_answer_length,
                    use_chars_length=self.use_chars_length,
                    text_postprocess=self.text_postprocess,
                )
                imap = p.imap_unordered(iter_func, raw_df.groupby("id"))
                res_sets = list(tqdm(imap, total=raw_df["id"].nunique()))
                p.close()
                p.join()
        else:
            res_sets = []
            for grp_pair in tqdm(raw_df.groupby("id")):
                res_sets.append(
                    _apply_extract_best_answer_pred(
                        grp_pair=grp_pair,
                        n_best_size=self.n_best_size,
                        max_answer_length=self.max_answer_length,
                        use_chars_length=self.use_chars_length,
                        text_postprocess=self.text_postprocess,
                    )
                )
        res_sets = sorted(res_sets)

        res_ids: List[str] = []
        res_answer_texts: List[str] = []
        res_answer_preds: List[str] = []
        for id, answer_text, answer_pred in res_sets:
            res_ids.append(id)
            res_answer_texts.append(answer_text)
            res_answer_preds.append(answer_pred)

        return res_ids, res_answer_texts, res_answer_preds


def _apply_extract_best_answer_pred(
    grp_pair: Tuple[str, DataFrame],
    n_best_size: int,
    max_answer_length,
    use_chars_length: bool,
    text_postprocess: Optional[str],
) -> Tuple[str, str, str]:
    id, grp_df = grp_pair
    answer_text = grp_df["answer_text"].iloc[0]
    if (grp_df["answer_text"] != answer_text).any():
        raise Exception(
            f"answer_texts are not same in the same id for {id}." f"df: {grp_df}"
        )
    answer_pred = _extract_best_answer_pred(
        contexts=grp_df["context"].tolist(),
        offset_mappings=grp_df["offset_mapping"].tolist(),
        start_logits=grp_df["start_logit"].tolist(),
        end_logits=grp_df["end_logit"].tolist(),
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
        use_chars_length=use_chars_length,
        text_postprocess=text_postprocess,
    )
    return id, answer_text, answer_pred


def _extract_best_answer_pred(
    contexts: List[str],
    offset_mappings: List[List[Tuple[int, int]]],
    start_logits: List[List[float]],
    end_logits: List[List[float]],
    n_best_size: int,
    max_answer_length: int,
    use_chars_length: bool,
    text_postprocess: Optional[str],
) -> str:
    candidates = []
    for context, offset_mapping, start_logit, end_logit in zip(
        contexts, offset_mappings, start_logits, end_logits
    ):
        candidates.extend(
            _extract_candidate_answer_preds(
                context=context,
                offset_mapping=offset_mapping,
                start_logit=np.asarray(start_logit),
                end_logit=np.asarray(end_logit),
                n_best_size=n_best_size,
                max_answer_length=max_answer_length,
                use_chars_length=use_chars_length,
            )
        )
    best_candidate = _choose_best_candidate(candidates=candidates)
    if text_postprocess == "baseline_kernel1":
        best_candidate = baseline_kernel1_text_postprocess(
            answer_text=best_candidate, context=contexts[0]
        )
    elif text_postprocess == "baseline_kernel2":
        best_candidate = baseline_kernel2_text_postprocess(
            answer_text=best_candidate, context=contexts[0]
        )
    elif text_postprocess == "baseline_kernel3":
        best_candidate = baseline_kernel3_text_postprocess(
            answer_text=best_candidate, context=contexts[0]
        )
    elif text_postprocess == "mypospro_ver1":
        best_candidate = mypospro_ver1(answer_text=best_candidate, context=contexts[0])
    elif text_postprocess == "mypospro_ver2":
        best_candidate = mypospro_ver2(answer_text=best_candidate, context=contexts[0])
    return best_candidate


def baseline_kernel1_text_postprocess(answer_text: str, context: str) -> str:
    bad_starts = [".", ",", "(", ")", "-", "–", ",", ";"]
    bad_endings = ["...", "-", "(", ")", "–", ",", ";"]

    tamil_ad = "கி.பி"
    tamil_bc = "கி.மு"
    tamil_km = "கி.மீ"
    hindi_ad = "ई"
    hindi_bc = "ई.पू"

    if answer_text == "":
        return answer_text
    while any([answer_text.startswith(y) for y in bad_starts]):
        answer_text = answer_text[1:]
    while any([answer_text.endswith(y) for y in bad_endings]):
        if answer_text.endswith("..."):
            answer_text = answer_text[:-3]
        else:
            answer_text = answer_text[:-1]
    if answer_text.endswith("..."):
        answer_text = answer_text[:-3]
    if (
        any(
            [
                answer_text.endswith(tamil_ad),
                answer_text.endswith(tamil_bc),
                answer_text.endswith(tamil_km),
                answer_text.endswith(hindi_ad),
                answer_text.endswith(hindi_bc),
            ]
        )
        and answer_text + "." in context
    ):
        answer_text = answer_text + "."
    return answer_text


def baseline_kernel2_text_postprocess(answer_text: str, context: str) -> str:
    answer_text = baseline_kernel1_text_postprocess(
        answer_text=answer_text, context=context
    )
    if re.search("[0-9]\.$", answer_text):
        answer_text = answer_text[:-1]
    return answer_text


def baseline_kernel3_text_postprocess(answer_text: str, context: str) -> str:
    answer_text = " ".join(answer_text.split())
    answer_text = answer_text.strip(punctuation)
    answer_text = baseline_kernel1_text_postprocess(
        answer_text=answer_text, context=context
    )
    return answer_text


def marathi_number_to_arabic_number(
    answer_text: str, context: str, only_whole_numeric: bool
) -> str:
    if only_whole_numeric and not answer_text.isnumeric():
        return answer_text

    marathi_to_arabic = {
        "०": "0",
        "१": "1",
        "२": "2",
        "३": "3",
        "४": "4",
        "५": "5",
        "६": "6",
        "७": "7",
        "८": "8",
        "९": "9",
    }

    new_answer_text = answer_text
    for marathi_num, arabic_num in marathi_to_arabic.items():
        new_answer_text = new_answer_text.replace(marathi_num, arabic_num)
    if context.find(new_answer_text) > 0:
        return new_answer_text
    return answer_text


def mypospro_ver1(answer_text: str, context: str) -> str:
    answer_text = " ".join(answer_text.split())
    answer_text = answer_text.strip(
        "".join(list(filter(lambda x: x != "%", punctuation)))
    )
    answer_text = baseline_kernel1_text_postprocess(
        answer_text=answer_text, context=context
    )
    answer_text = marathi_number_to_arabic_number(
        answer_text=answer_text, context=context, only_whole_numeric=True
    )
    return answer_text


def mypospro_ver2(answer_text: str, context: str) -> str:
    answer_text = " ".join(answer_text.split())
    answer_text = answer_text.strip(
        "".join(list(filter(lambda x: x != "%", punctuation)))
    )
    answer_text = baseline_kernel1_text_postprocess(
        answer_text=answer_text, context=context
    )
    answer_text = marathi_number_to_arabic_number(
        answer_text=answer_text, context=context, only_whole_numeric=False
    )
    return answer_text


def _extract_candidate_answer_preds(
    context: str,
    offset_mapping: List[Tuple[int, int]],
    start_logit: ndarray,
    end_logit: ndarray,
    n_best_size: int,
    max_answer_length: int,
    use_chars_length: bool,
) -> List[str]:
    start_indexes = np.argsort(start_logit)[-1 : -n_best_size - 1 : -1].tolist()
    end_indexes = np.argsort(end_logit)[-1 : -n_best_size - 1 : -1].tolist()

    candidates = []
    for start_index in start_indexes:
        for end_index in end_indexes:
            # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
            # to part of the input_ids that are not in the context.
            if (
                start_index >= len(offset_mapping)
                or end_index >= len(offset_mapping)
                or offset_mapping[start_index][0] == -1
                or offset_mapping[end_index][0] == -1
            ):
                continue

            start_char_index = offset_mapping[start_index][0]
            end_char_index = offset_mapping[end_index][1]
            extracted_text = context[start_char_index:end_char_index]
            # Don't consider answers with a length that is either < 0 or > max_answer_length.
            if end_index < start_index or (
                (use_chars_length and len(extracted_text) > max_answer_length)
                or (
                    not use_chars_length
                    and end_index - start_index + 1 > max_answer_length
                )
            ):
                continue

            score = float(start_logit[start_index] + end_logit[end_index])
            candidates.append({"extracted_text": extracted_text, "score": score})
    return candidates


def _choose_best_candidate(candidates: List[Dict[str, Any]]) -> str:
    if len(candidates) == 0:
        print("couldn't find best start end. use context as answer.")
        best_candidate = ""
    else:
        best_candidate = sorted(candidates, key=lambda x: x["score"], reverse=True)[0][
            "extracted_text"
        ]
    return best_candidate
