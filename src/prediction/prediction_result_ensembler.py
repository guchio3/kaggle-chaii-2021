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


class SimplePredictionResultEnsembler:
    def __init__(self, logger: myLogger) -> None:
        self.body: Dict[str, List[Union[str, List[Tuple[int, int]], Tensor]]] = {
            "ids": [],
            "offset_mappings": [],
            "start_logits": [],
            "end_logits": [],
        }
        self.logger = logger

    def add(
        self,
        ensemble_weight: float,
        id: str,
        simple_index: int,
        offset_mapping: List[Tuple[int, int]],
        start_logit: Tensor,
        end_logit: Tensor,
    ) -> None:
        if simple_index == len(self.body["ids"]):
            self.body["ids"].append(id)
            self.body["offset_mappings"].append(offset_mapping)
            self.body["start_logits"].append(start_logit)
            self.body["end_logits"].append(end_logit)
        elif simple_index < len(self.body["ids"]):
            self.body["start_logits"][simple_index] += ensemble_weight * start_logit
            self.body["end_logits"][simple_index] += ensemble_weight * end_logit
        else:
            raise Exception("not incremental index.")

    def to_prediction_result(self) -> PredictionResult:
        res_prediction_result = PredictionResult(ensemble_weight=0)
        for i in range(len(self.body["ids"])):
            id = self.body["ids"][i]
            offset_mapping = self.body["offset_mappings"][i]
            start_logit = self.body["start_logits"][i]
            end_logit = self.body["end_logits"][i]
            res_prediction_result.ids.append(id)
            res_prediction_result.offset_mappings.append(offset_mapping)
            res_prediction_result.start_logits.append(start_logit)
            res_prediction_result.end_logits.append(end_logit)
        return res_prediction_result


class PredictionResultEnsembler:
    def __init__(
        self, id_to_context_len: Dict[str, int], ensemble_mode: str, logger: myLogger
    ) -> None:
        valid_ensemble_modes = ["mean", "max"]
        if ensemble_mode not in valid_ensemble_modes:
            raise Exception(
                f"{ensemble_mode} is not supported."
                f"only {valid_ensemble_modes} are valid."
            )
        self.body: Dict[str, Dict[str, List[Union[int, float]]]] = {}
        self.id_to_context_len = id_to_context_len
        self.ensemble_mode = ensemble_mode
        self.logger = logger

    def add(
        self,
        ensemble_weight: float,
        id: str,
        offset_mapping: Tensor,  # List[Tuple[int, int]],
        start_logit: Tensor,
        end_logit: Tensor,
    ) -> None:
        if id not in self.body:
            id_context_len = self.id_to_context_len[id]
            if self.ensemble_mode == "mean":
                self.body[id] = {
                    "count": np.zeros(id_context_len),
                    "start_logit": np.zeros(id_context_len),
                    "end_logit": np.zeros(id_context_len),
                }
            elif self.ensemble_mode == "max":
                self.body[id] = {
                    "count": np.zeros(id_context_len),
                    "start_logit": np.ones(id_context_len) * -1_000_000,
                    "end_logit": np.zeros(id_context_len) * -1_000_000,
                }
            else:
                Exception()
        _add_operation(
            ensemble_weight=ensemble_weight,
            offset_mapping=offset_mapping.numpy().astype(int),  # .tolist()
            count=self.body[id]["count"],
            base_start_logit=self.body[id]["start_logit"],
            start_logit=start_logit.numpy(),
            base_end_logit=self.body[id]["end_logit"],
            end_logit=end_logit.numpy(),
            ensemble_mode=self.ensemble_mode,
        )

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
            # some chars are ignored, so count == 0. for ex, 6th of 22bff3dec
            if (count == 0).any().item():
                zero_cnt_index = torch.where(count == 0)
                max_count = int(count.max())
                count[zero_cnt_index] = max_count
                start_min_value = float(start_logit.min())
                start_logit[zero_cnt_index] = start_min_value
                end_min_value = float(end_logit.min())
                end_logit[zero_cnt_index] = end_min_value

            res_prediction_result.ids.append(id)
            id_context_len = self.id_to_context_len[id]
            res_prediction_result.offset_mappings.append(
                [(i, i + 1) for i in range(id_context_len)]
            )
            if self.ensemble_mode == "mean":
                res_prediction_result.start_logits.append(start_logit / count)
                res_prediction_result.end_logits.append(end_logit / count)
            elif self.ensemble_mode == "max":
                res_prediction_result.start_logits.append(start_logit)
                res_prediction_result.end_logits.append(end_logit)
        return res_prediction_result


@jit(nopython=True)
def _add_operation(
    ensemble_weight: float,
    offset_mapping: ndarray,
    count: ndarray,
    base_start_logit: ndarray,
    start_logit: ndarray,
    base_end_logit: ndarray,
    end_logit: ndarray,
    ensemble_mode: str,
) -> None:
    # ) -> Tuple[List[int], List[float], List[float], List[float]]:
    for i in range(len(offset_mapping)):
        offset_mapping_i = offset_mapping[i]
        s_i = offset_mapping_i[0]
        e_i = offset_mapping_i[1]
        # if s_i == -1:
        #     continue
        start_logit_i = start_logit[i]
        end_logit_i = end_logit[i]
        for j in range(s_i, e_i):
            count[j] += 1
            if ensemble_mode == "mean":
                base_start_logit[j] += ensemble_weight * start_logit_i
                base_end_logit[j] += ensemble_weight * end_logit_i
            elif ensemble_mode == "max":
                base_start_logit[j] = max(start_logit_i, base_start_logit[j])
                base_end_logit[j] = max(end_logit_i, base_end_logit[j])
            else:
                raise Exception()


def calc_id_to_context_len(df: DataFrame):
    id_to_context_len = {}
    for _, row in df.iterrows():
        id_to_context_len[str(row["id"])] = len(row["context"])
    return id_to_context_len


# def ensemble_prediction_results(
#     prediction_results: List[PredictionResult],
#     id_to_context_len: Dict[str, int],
#     logger: myLogger,
# ) -> PredictionResult:
#     prediction_result_ensembler = PredictionResultEnsembler(
#         id_to_context_len=id_to_context_len, logger=logger
#     )
#
#     logger.info("now ensembling ...")
#     for prediction_result in tqdm(prediction_results):
#         for i in range(len(prediction_result)):
#             (
#                 id,
#                 offset_mapping,
#                 start_logit,
#                 end_logit,
#                 segmentaton_logit,
#             ) = prediction_result.get(i)
#             prediction_result_ensembler.add(
#                 ensemble_weight=prediction_result.ensemble_weight,
#                 id=id,
#                 offset_mapping=offset_mapping,
#                 start_logit=start_logit,
#                 end_logit=end_logit,
#             )
#     res_prediction_result = prediction_result_ensembler.to_prediction_result()
#     res_prediction_result.sort_values_based_on_ids()
#     res_prediction_result.convert_elems_to_larger_level_as_possible()
#     return res_prediction_result


def ensemble_prediction_result(
    prediction_result_ensembler: Union[
        SimplePredictionResultEnsembler, PredictionResultEnsembler
    ],
    prediction_result: PredictionResult,
) -> None:
    print("now ensembling ...")
    prediction_result_len = len(prediction_result)
    for i in tqdm(range(prediction_result_len), total=prediction_result_len):
        (
            id,
            offset_mapping,
            start_logit,
            end_logit,
        ) = prediction_result.get(i)
        if isinstance(prediction_result_ensembler, PredictionResultEnsembler):
            prediction_result_ensembler.add(
                ensemble_weight=prediction_result.ensemble_weight,
                id=id,
                offset_mapping=offset_mapping,
                start_logit=start_logit,
                end_logit=end_logit,
            )
        elif isinstance(prediction_result_ensembler, SimplePredictionResultEnsembler):
            prediction_result_ensembler.add(
                ensemble_weight=prediction_result.ensemble_weight,
                id=id,
                simple_index=i,
                offset_mapping=offset_mapping,
                start_logit=start_logit,
                end_logit=end_logit,
            )
        else:
            raise Exception()
    print("done.")
