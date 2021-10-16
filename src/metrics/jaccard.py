from typing import List


def jaccard(text_true: str, text_pred: str):
    a = set(text_true.lower().split())
    b = set(text_pred.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def calc_jaccard_mean(text_trues: List[str], text_preds: List[str]) -> float:
    if len(text_trues) != len(text_preds):
        raise Exception(
            f"len(text_trues) != len(text_preds), {len(text_trues)} and {len(text_preds)}"
        )

    jaccards = []
    for text_true, text_pred in zip(text_trues, text_preds):
        jaccards.append(jaccard(text_true=text_true, text_pred=text_pred))
    return sum(jaccards) / len(jaccards)
