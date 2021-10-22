from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from torch import Tensor
from tqdm.auto import tqdm

from src.log import myLogger


class Postprocessor(metaclass=ABCMeta):
    def __init__(
        self, logger: myLogger, n_best_size: int = 20, max_answer_length: int = 30,
    ) -> None:
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length
        self.logger = logger

    @abstractmethod
    def __call__(
        self,
        ids: List[str],
        contexts: List[str],
        answer_texts: List[str],
        offset_mappings: List[List[Tuple[int, int]]],
        start_logits: Tensor,
        end_logits: Tensor,
        segmentation_logits: Tensor,
    ) -> List[str]:
        raise NotImplementedError()


class BaselineKernelPostprocessor(Postprocessor):
    def __call__(
        self,
        ids: List[str],
        contexts: List[str],
        answer_texts: List[str],
        offset_mappings: List[List[Tuple[int, int]]],
        start_logits: Tensor,
        end_logits: Tensor,
        segmentation_logits: Tensor,
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
                "len(ids): {len(ids)}, "
                "len(contexts): {len(contexts)}, "
                "len(answer_texts): {len(answer_texts)}, "
                "len(offset_mappings): {len(offset_mappings)}, "
                "len(start_logits): {len(start_logits)}, "
                "len(end_logits): {len(end_logits)}."
            )

        raw_df = DataFrame()
        raw_df["id"] = ids
        raw_df["context"] = contexts
        raw_df["answer_text"] = answer_texts
        raw_df["offset_mapping"] = offset_mappings
        raw_df["start_logit"] = start_logits.tolist()
        raw_df["end_logit"] = end_logits.tolist()
        raw_df["segmentation_logit"] = segmentation_logits.tolist()

        res_ids: List[str] = []
        res_answer_preds: List[str] = []
        res_answer_texts: List[str] = []
        for id, grp_df in tqdm(raw_df.groupby("id")):
            res_ids.append(id)
            answer_text = grp_df["answer_text"].iloc[0]
            if (grp_df["answer_text"] != answer_text).any():
                raise Exception(
                    f"answer_texts are not same in the same id for {id}."
                    f"df: {grp_df}"
                )
            res_answer_texts.append(answer_text)
            answer_pred = self._extract_best_answer_pred(
                contexts=grp_df["context"].tolist(),
                offset_mappings=grp_df["offset_mapping"].tolist(),
                start_logits=grp_df["start_logit"].tolist(),
                end_logits=grp_df["end_logit"].tolist(),
            )
            res_answer_preds.append(answer_pred)
        return res_ids, res_answer_texts, res_answer_preds

    def _extract_best_answer_pred(
        self,
        contexts: List[str],
        offset_mappings: List[List[Tuple[int, int]]],
        start_logits: List[List[float]],
        end_logits: List[List[float]],
    ) -> str:
        candidates = []
        for context, offset_mapping, start_logit, end_logit in zip(
            contexts, offset_mappings, start_logits, end_logits
        ):
            candidates.extend(
                self._extract_candidate_answer_preds(
                    context=context,
                    offset_mapping=offset_mapping,
                    start_logit=np.asarray(start_logit),
                    end_logit=np.asarray(end_logit),
                )
            )
        best_candidate = self._choose_best_candidate(candidates=candidates)
        return best_candidate

    def _extract_candidate_answer_preds(
        self,
        context: str,
        offset_mapping: List[Tuple[int, int]],
        start_logit: ndarray,
        end_logit: ndarray,
    ) -> List[str]:
        start_indexes = np.argsort(start_logit.numpy())[
            -1 : -self.n_best_size - 1 : -1
        ].tolist()
        end_indexes = np.argsort(end_logit.numpy())[
            -1 : -self.n_best_size - 1 : -1
        ].tolist()

        candidates = []
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                # to part of the input_ids that are not in the context.
                if (
                    start_index >= len(offset_mapping)
                    or end_index >= len(offset_mapping)
                    or offset_mapping[start_index] is None
                    or offset_mapping[end_index] is None
                ):
                    continue
                # Don't consider answers with a length that is either < 0 or > max_answer_length.
                if (
                    end_index < start_index
                    or end_index - start_index + 1 > self.max_answer_length
                ):
                    continue

                start_char = offset_mapping[start_index][0]
                end_char = offset_mapping[end_index][1]

                extracted_text = context[start_char:end_char]
                score = float(start_logit[start_index] + end_logit[end_index])
                candidates.append({"extracted_text": extracted_text, "score": score})
        return candidates

    def _choose_best_candidate(self, candidates: List[Dict[str, Any]]) -> str:
        if len(candidates) == 0:
            self.logger.warn("couldn't find best start end. use context as answer.")
            best_candidate = ""
        else:
            best_candidate = sorted(candidates, key=lambda x: x["score"], reverse=True)[
                0
            ]["extracted_text"]
        return best_candidate
