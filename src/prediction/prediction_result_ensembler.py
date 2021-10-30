from typing import List

from tqdm.auto import tqdm

from src.log import myLogger
from src.prediction.prediction_result import PredictionResult
from src.timer import class_dec_timer


class PredictionResultEnsembler:
    def __init__(self, logger: myLogger) -> None:
        self.logger = logger

    @class_dec_timer(unit="m")
    def ensemble(self, prediction_results: List[PredictionResult]) -> PredictionResult:
        res_prediction_result = PredictionResult(ensemble_weight=0)
        self.logger.info(f"now ensembling ...")
        for prediction_result in tqdm(prediction_results):
            prediction_result.convert_elems_to_char_level()
            prediction_result.sort_values_based_on_ids()
            if len(res_prediction_result) == 0:
                res_prediction_result.ids = prediction_result.ids
                res_prediction_result.offset_mappings = (
                    prediction_result.offset_mappings
                )
                res_prediction_result.start_logits = prediction_result.start_logits
                res_prediction_result.end_logits = prediction_result.end_logits
                res_prediction_result.segmentation_logits = (
                    prediction_result.segmentation_logits
                )
            else:
                if res_prediction_result.ids != prediction_result.ids:
                    raise Exception(
                        "res_prediction_result.ids != prediction_result.ids"
                    )
                for i in range(len(prediction_result)):
                    res_prediction_result.start_logits[i] += (
                        prediction_result.ensemble_weight
                        * prediction_result.start_logits[i]
                    )
                    res_prediction_result.end_logits[i] += (
                        prediction_result.ensemble_weight
                        * prediction_result.end_logits[i]
                    )
                    res_prediction_result.segmentation_logits[i] += (
                        prediction_result.ensemble_weight
                        * prediction_result.segmentation_logits[i]
                    )
        res_prediction_result.convert_elems_to_larger_level_as_possible()
        return res_prediction_result
