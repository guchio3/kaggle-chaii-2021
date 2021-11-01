from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from numba import jit
from numpy import ndarray
from pandas import DataFrame
from torch import Tensor
from tqdm.auto import tqdm

from src.log import myLogger
from src.prediction.prediction_result import PredictionResult
from src.timer import dec_timer


class PredictionResultEnsembler:
    def __init__(self, id_to_context_len: Dict[str, int], logger: myLogger) -> None:
        self.body: Dict[str, Dict[str, List[Union[int, float]]]] = {}
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
                "count": np.zeros(id_context_len),
                "start_logit": np.zeros(id_context_len),
                "end_logit": np.zeros(id_context_len),
                "segmentation_logit": np.zeros(id_context_len),
            }
        _add_operation(
            ensemble_weight=ensemble_weight,
            offset_mapping=offset_mapping.numpy(),  # .tolist()
            count=self.body[id]["count"],
            base_start_logit=self.body[id]["start_logit"],
            start_logit=start_logit.numpy(),
            base_end_logit=self.body[id]["end_logit"],
            end_logit=end_logit.numpy(),
            base_segmentation_logit=self.body[id]["segmentation_logit"],
            segmentation_logit=segmentation_logit.numpy(),
        )

    # def add(
    #     self,
    #     ensemble_weight: float,
    #     id: str,
    #     offset_mapping: List[Tuple[int, int]],
    #     start_logit: Tensor,
    #     end_logit: Tensor,
    #     segmentation_logit: Tensor,
    # ) -> None:
    #     if id not in self.body:
    #         id_context_len = self.id_to_context_len[id]
    #         self.body[id] = {
    #             "count": torch.zeros(id_context_len),
    #             "start_logit": torch.zeros(id_context_len),
    #             "end_logit": torch.zeros(id_context_len),
    #             "segmentation_logit": torch.zeros(id_context_len),
    #         }
    #     for (s_i, e_i), start_logit_i, end_logit_i, segmentation_logit_i in zip(
    #         offset_mapping, start_logit, end_logit, segmentation_logit
    #     ):
    #         if s_i == -1:
    #             continue
    #         for j in range(s_i, e_i):
    #             self.body[id]["count"][j] += 1
    #             self.body[id]["start_logit"][j] += ensemble_weight * start_logit_i
    #             self.body[id]["end_logit"][j] += ensemble_weight * end_logit_i
    #             self.body[id]["segmentation_logit"][j] += (
    #                 ensemble_weight * segmentation_logit_i
    #             )

    def to_prediction_result(self) -> PredictionResult:
        # temptemptemptemp
        temp_key = list(self.body.keys())[0]
        self.logger.info(f"{self.body[temp_key]['count']}")
        # temptemptemptemp
        res_prediction_result = PredictionResult(ensemble_weight=0)
        for id in self.body.keys():
            count = Tensor(self.body[id]["count"])
            start_logit = Tensor(self.body[id]["start_logit"])
            end_logit = Tensor(self.body[id]["end_logit"])
            segmentation_logit = Tensor(self.body[id]["segmentation_logit"])
            # some chars are ignored, so count == 0. for ex, 6th of 22bff3dec
            if (count == 0).any().item():
                zero_cnt_index = torch.where(count == 0)
                max_count = int(count.max())
                count[zero_cnt_index] = max_count
                start_min_value = float(start_logit.min())
                start_logit[zero_cnt_index] = start_min_value
                end_min_value = float(end_logit.min())
                end_logit[zero_cnt_index] = end_min_value
                segmentation_min_value = float(segmentation_logit.min())
                segmentation_logit[zero_cnt_index] = segmentation_min_value

            res_prediction_result.ids.append(id)
            id_context_len = self.id_to_context_len[id]
            res_prediction_result.offset_mappings.append(
                [(i, i + 1) for i in range(id_context_len)]
            )
            res_prediction_result.start_logits.append(start_logit / count)
            res_prediction_result.end_logits.append(end_logit / count)
            res_prediction_result.segmentation_logits.append(segmentation_logit / count)
        return res_prediction_result


# @jit(nopython=True)
# def _add_operation(
#     ensemble_weight: float,
#     offset_mapping: ndarray,  # List[Tuple[int, int]],
#     count: List[int],
#     base_start_logit: List[float],
#     start_logit: List[float],
#     base_end_logit: List[float],
#     end_logit: List[float],
#     base_segmentation_logit: List[float],
#     segmentation_logit: List[float],
# ) -> None:
@jit(nopython=True)
def _add_operation(
    ensemble_weight: float,
    offset_mapping: ndarray,
    count: ndarray,
    base_start_logit: ndarray,
    start_logit: ndarray,
    base_end_logit: ndarray,
    end_logit: ndarray,
    base_segmentation_logit: ndarray,
    segmentation_logit: ndarray,
) -> None:
    # ) -> Tuple[List[int], List[float], List[float], List[float]]:
    for i in range(len(offset_mapping)):
        offset_mapping_i = offset_mapping[i]
        s_i = offset_mapping_i[0]
        e_i = offset_mapping_i[1]
        if s_i == -1:
            continue
        start_logit_i = start_logit[i]
        end_logit_i = end_logit[i]
        segmentation_logit_i = segmentation_logit[i]
        for j in range(s_i, e_i):
            count[j] += 1
            base_start_logit[j] += ensemble_weight * start_logit_i
            base_end_logit[j] += ensemble_weight * end_logit_i
            base_segmentation_logit[j] += ensemble_weight * segmentation_logit_i


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
    logger.info("now ensembling ...")
    for prediction_results in tqdm(prediction_results):
        for i in range(len(prediction_results)):
            (
                id,
                offset_mapping,
                start_logit,
                end_logit,
                segmentaton_logit,
            ) = prediction_results.get(i)
            prediction_result_ensembler.add(
                ensemble_weight=prediction_results.ensemble_weight,
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


@dec_timer(unit="m")
def ensemble_prediction_result(
    prediction_result_ensembler: PredictionResultEnsembler,
    prediction_result: PredictionResult,
    logger: myLogger,
) -> None:
    logger.info("now ensembling ...")
    for prediction_result in tqdm(prediction_result):
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
