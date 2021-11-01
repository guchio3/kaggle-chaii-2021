import numpy as np
from numba.typed import List as numbaList

from src.prediction.prediction_result_ensembler import _add_operation


class TestPredictionResult:
    def test_add_operation(self) -> None:
        NUM = 100
        # offset_mapping = numbaList([(i, i + 1) for i in range(NUM)])
        # count = numbaList([0] * NUM)
        # base_start_logit = numbaList([0.0] * NUM)
        # start_logit = numbaList([1.0] * NUM)
        # base_end_logit = numbaList([0.0] * NUM)
        # end_logit = numbaList([1.0] * NUM)
        # base_segmentation_logit = numbaList([0.0] * NUM)
        # segmentation_logit = numbaList([1.0] * NUM)
        offset_mapping = [(i, i + 1) for i in range(NUM)]
        count = np.zeros(NUM) # [0] * NUM
        print("-------------------------------")
        print(count)
        print("-------------------------------")
        base_start_logit = [0.0] * NUM
        start_logit = [1.0] * NUM
        base_end_logit = [0.0] * NUM
        end_logit = [1.0] * NUM
        base_segmentation_logit = [0.0] * NUM
        segmentation_logit = [1.0] * NUM
        _add_operation(
            ensemble_weight=10,
            offset_mapping=np.asarray(offset_mapping),
            count=count,
            base_start_logit=np.asarray(base_start_logit),
            start_logit=np.asarray(start_logit),
            base_end_logit=np.asarray(base_end_logit),
            end_logit=np.asarray(end_logit),
            base_segmentation_logit=np.asarray(base_segmentation_logit),
            segmentation_logit=np.asarray(segmentation_logit),
        )
        assert _add_operation.nopython_signatures
        assert (np.asarray(count) == 1).all()
        assert (np.asarray(base_start_logit) == 10).all()
        assert (np.asarray(base_end_logit) == 10).all()
        assert (np.asarray(base_segmentation_logit) == 10).all()
