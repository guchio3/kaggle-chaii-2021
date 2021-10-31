from typing import Dict, List, Tuple

import torch
from pandas import DataFrame
from torch import Tensor
from tqdm.auto import tqdm

from src.log import myLogger
from src.prediction.prediction_result import PredictionResult


def calc_id_to_context_len(df: DataFrame):
    id_to_context_len = {}
    for _, row in df.iterrows():
        id_to_context_len[str(row["id"])] = len(row["context"])
    return id_to_context_len


def ensemble_prediction_results(
    prediction_results: List[PredictionResult],
    id_to_context_len: Dict[str, int],
    logger: myLogger,
) -> PredictionResult:
    prediction_result_ensembler = PredictionResultEnsembler(
        id_to_context_len=id_to_context_len, logger=logger
    )

    # res_prediction_result = PredictionResult(ensemble_weight=0)
    logger.info(f"now ensembling ...")
    for prediction_result in tqdm(prediction_results):
        for i in range(len(prediction_result)):
            (
                id,
                offset_mapping,
                start_logit,
                end_logit,
                segmentaton_logit,
            ) = prediction_result.get(i)
            prediction_result_ensembler.add(
                ensemble_weight=prediction_result.ensemble_weight,
                id=id,
                offset_mapping=offset_mapping,
                start_logit=start_logit,
                end_logit=end_logit,
                segmentation_logit=segmentaton_logit,
            )
        # ##### prediction_result.convert_elems_to_char_level()
        # ##### prediction_result.sort_values_based_on_ids()
        # if len(res_prediction_result) == 0:
        #     logger.info("len = 0")
        #     res_prediction_result.ids = prediction_result.ids
        #     res_prediction_result.offset_mappings = prediction_result.offset_mappings
        #     res_prediction_result.start_logits = prediction_result.start_logits
        #     res_prediction_result.end_logits = prediction_result.end_logits
        #     res_prediction_result.segmentation_logits = (
        #         prediction_result.segmentation_logits
        #     )
        # else:
        #     logger.info(f"len = {len(res_prediction_result)}")
        #     if res_prediction_result.ids != prediction_result.ids:
        #         raise Exception("res_prediction_result.ids != prediction_result.ids")
        #     for i in range(len(prediction_result)):
        #         res_prediction_result.start_logits[i] += (
        #             prediction_result.ensemble_weight
        #             * prediction_result.start_logits[i]
        #         )
        #         res_prediction_result.end_logits[i] += (
        #             prediction_result.ensemble_weight * prediction_result.end_logits[i]
        #         )
        #         res_prediction_result.segmentation_logits[i] += (
        #             prediction_result.ensemble_weight
        #             * prediction_result.segmentation_logits[i]
        #         )
    res_prediction_result = prediction_result_ensembler.to_prediction_result()
    res_prediction_result.sort_values_based_on_ids()
    res_prediction_result.convert_elems_to_larger_level_as_possible()
    return res_prediction_result


class PredictionResultEnsembler:
    def __init__(self, id_to_context_len: Dict[str, int], logger: myLogger) -> None:
        self.body: Dict[str, Dict[str, Tensor]] = {}
        self.id_to_context_len = id_to_context_len
        self.logger = logger

    def add(
        self,
        ensemble_weight: float,
        id: str,
        offset_mapping: List[Tuple[int, int]],
        start_logit: Tensor,
        end_logit: Tensor,
        segmentation_logit: Tensor,
    ) -> None:
        if id not in self.body:
            id_context_len = self.id_to_context_len[id]
            self.body[id] = {
                "start_logit": torch.zeros(id_context_len),
                "end_logit": torch.zeros(id_context_len),
                "segmentation_logit": torch.zeros(id_context_len),
                "count": torch.zeros(id_context_len),
            }
        for (s_i, e_i), start_logit_i, end_logit_i, segmentation_logit_i in zip(
            offset_mapping, start_logit, end_logit, segmentation_logit
        ):
            for j in range(s_i, e_i):
                self.body[id]["start_logit"][j] += ensemble_weight * start_logit_i
                self.body[id]["end_logit"][j] += ensemble_weight * end_logit_i
                self.body[id]["segmentation_logit"][j] += (
                    ensemble_weight * segmentation_logit_i
                )
                self.body[id]["count"][j] += 1

    def to_prediction_result(self) -> PredictionResult:
        # temptemptemptemp
        temp_key = list(self.body.keys())[0]
        self.logger.info(f"{self.body[temp_key]['count']}")
        # temptemptemptemp
        res_prediction_result = PredictionResult(ensemble_weight=0)
        for id in self.body.keys():
            count = self.body[id]["count"]
            if (count == 0).any().item():
                raise Exception(f"count contains 0, {count}")

            res_prediction_result.ids.append(id)
            id_context_len = self.id_to_context_len[id]
            res_prediction_result.offset_mappings.append(
                [(i, i + 1) for i in range(id_context_len)]
            )
            res_prediction_result.start_logits.append(
                self.body[id]["start_logit"] / count
            )
            res_prediction_result.end_logits.append(self.body[id]["end_logit"] / count)
            res_prediction_result.segmentation_logits.append(
                self.body[id]["segmentation_logit"] / count
            )
        return res_prediction_result
