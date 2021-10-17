from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import numpy as np
from torch import Tensor

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
        contexts: List[str],
        offset_mappings: List[List[Tuple[int, int]]],
        start_logits: Tensor,
        end_logits: Tensor,
        segmentation_logits: Tensor,
    ) -> List[str]:
        raise NotImplementedError()


class BaselineKernelPostprocessor(Postprocessor):
    def __call__(
        self,
        contexts: List[str],
        offset_mappings: List[List[Tuple[int, int]]],
        start_logits: Tensor,
        end_logits: Tensor,
        segmentation_logits: Tensor,
    ) -> List[str]:
        if (
            len(contexts) != len(offset_mappings)
            or len(offset_mappings) != len(start_logits)
            or len(start_logits) != len(end_logits)
        ):
            raise Exception(
                "len of contexts, offset_mappings, start_logits, and end_logits are different. "
                "len(contexts): {len(contexts)}, "
                "len(offset_mappings): {len(offset_mappings)}, "
                "len(start_logits): {len(start_logits)}, "
                "len(end_logits): {len(end_logits)}."
            )
        pred_texts = []
        for context, offset_mapping, start_logit, end_logit in zip(
            contexts, offset_mappings, start_logits, end_logits
        ):
            pred_text = self._extract_text(
                context=context,
                offset_mapping=offset_mapping,
                start_logit=start_logit,
                end_logit=end_logit,
            )
            pred_texts.append(pred_text)
        return pred_texts

    def _extract_text(
        self,
        context: str,
        offset_mapping: List[Tuple[int, int]],
        start_logit: Tensor,
        end_logit: Tensor,
    ) -> str:
        start_indexes = np.argsort(start_logit)[
            -1 : -self.n_best_size - 1 : -1
        ].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -self.n_best_size - 1 : -1].tolist()
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
                break
        else:
            raise Exception("couldn't find extracted_text")

        return extracted_text
