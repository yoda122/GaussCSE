from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from tqdm import tqdm

from src import utils
from src.models import GaussOutput
from src.similarity import asymmetrical_kl_sim


class DirectionPredictionEvaluator:
    def __init__(
        self,
        dataset_dir: Path,
        dist_fn: Callable[[list[str], list[str]], GaussOutput],
    ) -> None:
        self.dataset_dir = dataset_dir
        self.dist_fn = dist_fn

        self.datasets = {
            "snli": self.load_df("snli/test.jsonl"),
            "mnli": self.load_df("mnli/test.jsonl"),
            "sick": self.load_df("sick/test.jsonl"),
        }

    def load_df(self, path: str) -> pd.DataFrame:
        df = utils.load_jsonl(self.dataset_dir / path)
        df = df[df["label"] == "entailment"]
        return df

    @torch.inference_mode()
    def eval(self) -> dict[str, dict[str, float]]:
        results = {}

        for name, df in tqdm(
            list(self.datasets.items()),
            dynamic_ncols=True,
            leave=False,
            desc="Direction",
        ):
            results[name] = self.run(df)
        return results

    @torch.inference_mode()
    def run(self, df: pd.DataFrame) -> dict[str, float]:
        premise: GaussOutput = self.dist_fn(df["premise"].tolist())
        hypothesis: GaussOutput = self.dist_fn(df["hypothesis"].tolist())

        return {
            "sim_acc": self.sim_acc(premise, hypothesis),
            "det_acc": self.det_acc(premise, hypothesis),
        }

    def sim_acc(
        self,
        premise: GaussOutput,
        hypothesis: GaussOutput,
    ) -> dict[str, float]:
        sim_for = asymmetrical_kl_sim(
            hypothesis.mu, hypothesis.std, premise.mu, premise.std
        )
        sim_rev = asymmetrical_kl_sim(
            premise.mu, premise.std, hypothesis.mu, hypothesis.std
        )
        sub = sim_for - sim_rev
        acc = (sub > 0).float().sum().item() / sub.size(0)
        return acc

    def det_acc(
        self,
        premise: GaussOutput,
        hypothesis: GaussOutput,
    ) -> dict[str, float]:
        det1 = torch.log(premise.std).sum(dim=-1)
        det2 = torch.log(hypothesis.std).sum(dim=-1)
        sub = det1 - det2
        acc = (sub > 0).float().sum().item() / sub.size(0)
        return acc
