from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from sklearn.metrics import auc as calc_auc
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

from src import utils


class NLIEvaluator:
    def __init__(
        self,
        dataset_dir: Path,
        sim_fn: Callable[[list[str], list[str]], list[float]],
    ) -> None:
        self.dataset_dir = dataset_dir
        self.sim_fn = sim_fn

        self.datasets = {
            "snli": self.load_df("snli/test.jsonl"),
            "mnli": self.load_df("mnli/test.jsonl"),
            "sick": self.load_df("sick/test.jsonl"),
        }
        self.dev_df = self.load_df("snli/val.jsonl")

    def load_df(self, path: str) -> pd.DataFrame:
        df = utils.load_jsonl(self.dataset_dir / path)
        df["label"] = (df["label"] == "entailment").astype(int)
        return df

    @torch.inference_mode()
    def eval(self) -> dict[str, dict[str, float]]:
        dev_acc, threshold = self.dev_acc_threshold()
        dev_auc = self.dev_auc()

        results = {
            "dev": {
                "acc": dev_acc,
                "auc": dev_auc,
                "threshold": threshold,
            },
        }

        for name, df in tqdm(
            list(self.datasets.items()),
            dynamic_ncols=True,
            leave=False,
            desc="NLI",
        ):
            results[name] = self.run(df, threshold)
        return results

    @torch.inference_mode()
    def run(
        self,
        df: pd.DataFrame,
        threshold: float,
    ) -> dict[str, float]:
        y_score = self.sim_fn(df["hypothesis"].tolist(), df["premise"].tolist())
        return {
            "acc": self.acc(df["label"], y_score, threshold),
            "auc": self.auc(df["label"], y_score),
        }

    @torch.inference_mode()
    def dev_auc(self) -> float:
        y_score = self.sim_fn(
            self.dev_df["hypothesis"].tolist(), self.dev_df["premise"].tolist()
        )
        return self.auc(self.dev_df["label"], y_score)

    @torch.inference_mode()
    def dev_acc_threshold(self) -> tuple[float, float]:
        y_score = self.sim_fn(
            self.dev_df["hypothesis"].tolist(), self.dev_df["premise"].tolist()
        )
        return optimal_threshold_acc(self.dev_df["label"], y_score)

    def auc(self, y_true: list[int], y_score: list[float]) -> float:
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return calc_auc(recall, precision)

    def acc(
        self,
        y_true: list[int],
        y_score: list[float],
        threshold: float,
    ) -> float:
        tp, tn = 0, 0
        for label, score in zip(y_true, y_score):
            if score >= threshold and label == 1:
                tp += 1
            elif score < threshold and label == 0:
                tn += 1
        return (tp + tn) / len(y_true)


# c.f. https://stackoverflow.com/questions/30717688/how-to-compute-optimal-threshold-for-accuracy
def optimal_threshold_acc(y_true: list[int], y_score: list[float], num=1000):
    label_score = sorted(zip(y_true, y_score), key=lambda x: x[1])

    tp = sum(l for l, _ in label_score)
    tn = 0
    total = len(label_score)

    best_acc = (tp + tn) / total
    best_threshold = 0

    position = 0
    for i in range(0, num + 1):
        threshold = i / num
        label, score = label_score[position]
        if threshold <= score:
            continue

        if label == 1:
            tp -= 1
        else:
            tn += 1

        if position + 1 < len(label_score):
            _, next_score = label_score[position + 1]
            while threshold > next_score:
                position += 1
                label, score = label_score[position]
                if label == 1:
                    tp -= 1
                else:
                    tn += 1
                if position + 1 >= len(label_score):
                    break
                _, next_score = label_score[position + 1]

        acc = (tp + tn) / total

        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold

        position += 1
        if position >= len(label_score):
            break

    return best_acc, best_threshold


if __name__ == "__main__":
    y_true = [1, 0, 0, 0, 1, 1, 1, 0]
    y_score = [0.1, 0.2, 0.3, 0.349, 0.4, 0.7, 0.8, 0.9]

    # 4/8, 3/8, 4/8, 5/8, 6/8, 5/8, 4/8, 3/8, 4/8
    best_acc, threshold = optimal_threshold_acc(y_true, y_score)
    print(best_acc, threshold)
    assert best_acc == 0.75
    assert threshold == 0.35, threshold

    # 4/8, 3/8, 4/8, 6/8, 5/8, 4/8, 3/8, 4/8
    best_acc, threshold = optimal_threshold_acc(y_true, y_score, num=10)
    print(best_acc, threshold)
    assert best_acc == 0.75, best_acc
    assert threshold == 0.4, threshold

    # 4/8, 4/8, 5/8, 5/8, 3/8, 4/8
    best_acc, threshold = optimal_threshold_acc(y_true, y_score, num=5)
    print(best_acc, threshold)
    assert best_acc == 0.75, best_acc
    assert threshold == 0.4, threshold

    y_true = [0, 0, 0, 0]
    y_score = [0.10, 0.11, 0.12, 0.13]
    best_acc, threshold = optimal_threshold_acc(y_true, y_score, num=5)
    print(best_acc, threshold)
    assert best_acc == 1.0, best_acc
    assert threshold == 0.20, threshold

    y_true = [1, 1, 1, 1, 0, 1]
    y_score = [0.84, 0.83, 0.82, 0.81, 0.1, 0.2]
    best_acc, threshold = optimal_threshold_acc(y_true, y_score, num=5)
    print(best_acc, threshold)
    assert best_acc == 1.0, best_acc
    assert threshold == 0.20, threshold
