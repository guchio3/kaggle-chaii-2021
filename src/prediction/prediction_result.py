import gc
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
from torch import Tensor


@dataclass
class PredictionResult:
    ensemble_weight: float
    ids: List[str] = field(default_factory=list)
    offset_mappings: List[List[Tuple[int, int]]] = field(default_factory=list)
    start_logits: List[Tensor] = field(default_factory=list)
    end_logits: List[Tensor] = field(default_factory=list)

    def __len__(self) -> int:
        if (
            len(self.ids) != len(self.offset_mappings)
            or len(self.offset_mappings) != len(self.start_logits)
            or len(self.start_logits) != len(self.end_logits)
        ):
            raise Exception(
                "len of elems are different. "
                f"ids: {len(self.ids)} "
                f"offset_mappings: {len(self.offset_mappings)} "
                f"start_logits: {len(self.start_logits)} "
                f"end_logits: {len(self.end_logits)} "
            )
        return len(self.ids)

    def get(self, i: int) -> Tuple[str, List[Tuple[int, int]], Tensor, Tensor]:
        return (
            self.ids[i],
            self.offset_mappings[i],
            self.start_logits[i],
            self.end_logits[i],
        )

    def extend_by_value_list(self, key: str, value_list: Optional[List[str]]) -> None:
        if value_list is not None:
            getattr(self, key).extend(value_list)

    def extend_by_tensor(self, key: str, val_info: Optional[Tensor]) -> None:
        if val_info is not None:
            val_info = val_info.to("cpu")
            getattr(self, key).extend([val_info_i for val_info_i in val_info])
            # getattr(self, key).extend(val_info.tolist())

    # def convert_elems_to_char_level(self) -> None:
    #     new_offset_mappings = []
    #     new_start_logits = []
    #     new_end_logits = []
    #     new_segmentation_logits = []

    #     for i in range(len(self.ids)):
    #         (
    #             new_offset_mapping,
    #             new_start_logit,
    #             new_end_logit,
    #             new_segmentation_logit,
    #         ) = self._convert_elem_to_char_level(
    #             offset_mapping=self.offset_mappings[i],
    #             start_logit=self.start_logits[i],
    #             end_logit=self.end_logits[i],
    #             segmentation_logit=self.segmentation_logits[i],
    #         )
    #         new_offset_mappings.append(new_offset_mapping)
    #         new_start_logits.append(new_start_logit)
    #         new_end_logits.append(new_end_logit)
    #         new_segmentation_logits.append(new_segmentation_logit)

    #     self.offset_mappings = new_offset_mappings
    #     self.start_logits = new_start_logits
    #     self.end_logits = new_end_logits
    #     self.segmentation_logits = new_segmentation_logits

    # def _convert_elem_to_char_level(
    #     self,
    #     offset_mapping: List[Tuple[int, int]],
    #     start_logit: Tensor,
    #     end_logit: Tensor,
    #     segmentation_logit: Tensor,
    # ) -> Tuple[List[Tuple[int, int]], Tensor, Tensor, Tensor]:
    #     new_offset_mapping = []
    #     new_start_logit = []
    #     new_end_logit = []
    #     new_segmentation_logit = []
    #     for i, (s, e) in enumerate(offset_mapping):
    #         if s == -1:
    #             continue
    #         for j in range(s, e):
    #             new_offset_mapping.append((j, j + 1))
    #             new_start_logit.append(start_logit[i])
    #             new_end_logit.append(end_logit[i])
    #             new_segmentation_logit.append(segmentation_logit[i])
    #     return (
    #         new_offset_mapping,
    #         Tensor(new_start_logit),
    #         Tensor(new_end_logit),
    #         Tensor(new_segmentation_logit),
    #     )

    def sort_values_based_on_ids(self) -> None:
        new_ids = []
        new_offset_mappings = []
        new_start_logits = []
        new_end_logits = []

        for i in np.argsort(self.ids):
            new_ids.append(self.ids[i])
            new_offset_mappings.append(self.offset_mappings[i])
            new_start_logits.append(self.start_logits[i])
            new_end_logits.append(self.end_logits[i])

        self.ids = new_ids
        self.offset_mappings = new_offset_mappings
        self.start_logits = new_start_logits
        self.end_logits = new_end_logits

    def convert_elems_to_larger_level_as_possible(self) -> None:
        new_offset_mappings = []
        new_start_logits = []
        new_end_logits = []

        for i in range(len(self.ids)):
            (
                new_offset_mapping,
                new_start_logit,
                new_end_logit,
            ) = self._convert_elems_to_as_larger_level_as_possible(
                offset_mapping=self.offset_mappings[i],
                start_logit=self.start_logits[i],
                end_logit=self.end_logits[i],
            )
            new_offset_mappings.append(new_offset_mapping)
            new_start_logits.append(new_start_logit)
            new_end_logits.append(new_end_logit)

        self.offset_mappings = new_offset_mappings
        self.start_logits = new_start_logits
        self.end_logits = new_end_logits

    def _convert_elems_to_as_larger_level_as_possible(
        self,
        offset_mapping: List[Tuple[int, int]],
        start_logit: Tensor,
        end_logit: Tensor,
    ) -> Tuple[List[Tuple[int, int]], Tensor, Tensor]:
        new_offset_mapping = []
        new_start_logit = []
        new_end_logit = []

        cur_s, cur_e = offset_mapping[0]
        bef_start_logit_i = start_logit[0]
        bef_end_logit_i = end_logit[0]
        for (s, e), start_logit_i, end_logit_i in zip(
            offset_mapping[1:],
            start_logit[1:],
            end_logit[1:],
        ):
            if start_logit_i != bef_start_logit_i:
                new_offset_mapping.append((cur_s, cur_e))
                new_start_logit.append(bef_start_logit_i)
                new_end_logit.append(bef_end_logit_i)

                cur_s = s
                bef_start_logit_i = start_logit_i
                bef_end_logit_i = end_logit_i

            cur_e = e
        else:
            new_offset_mapping.append((cur_s, cur_e))
            new_start_logit.append(bef_start_logit_i)
            new_end_logit.append(bef_end_logit_i)

        return (
            new_offset_mapping,
            Tensor(new_start_logit),
            Tensor(new_end_logit),
        )

    def to_textbatched(self, max_length: int, stride: int) -> None:
        if max_length <= stride:
            raise Exception("max_length <= stride, so cannot textbatched.")
        new_ids = []
        new_offset_mappings = []
        new_start_logits = []
        new_end_logits = []

        for id, offset_mapping, start_logit, end_logit in zip(
            self.ids, self.offset_mappings, self.start_logits, self.end_logits
        ):
            s = 0
            e = s + max_length
            while s < len(offset_mapping):
                new_ids.append(id)
                new_offset_mappings.append(offset_mapping[s:e])
                new_start_logits.append(start_logit[s:e])
                new_end_logits.append(end_logit[s:e])

                s = e - stride
                e = s + max_length

        self.ids = new_ids
        self.offset_mappings = new_offset_mappings
        self.start_logits = new_start_logits
        self.end_logits = new_end_logits
        gc.collect()

    def weight_logits(self, weights: Union[Tensor, List[Tensor]]) -> None:
        if len(weights) != len(self):
            raise Exception("len(weights) != len(self)")
        for i in range(len(self)):
            weight = weights[i]
            self.start_logits[i] *= weight
            self.end_logits[i] *= weight
