import pandas as pd

from src.preprocessor.preprocessor import _prep_text_v1
from transformers import AutoTokenizer


class TestPreprocessor:
    def test_prep_text_v1(self) -> None:
        row = pd.Series(
            {
                "context": "I have a dream. Please call me later.",
                "question": "What do I have?",
                "answer_start": 9,
                "answer_text": "dream",
            }
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "./data/dataset/deepset/xlm-roberta-large-squad2/"
        )
        _, res_series = _prep_text_v1(
            row_pair=(0, row), tokenizer=tokenizer, max_length=384, is_test=False
        )
        assert res_series["start_position"] == 4
        assert res_series["end_position"] == 4
        # import pdb; pdb.set_trace()
        print(res_series)
