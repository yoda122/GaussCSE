from pathlib import Path
from typing import Callable

import torch
from sklearn.metrics.pairwise import paired_cosine_distances

from src.evals.direction import DirectionPredictionEvaluator
from src.evals.nli import NLIEvaluator
from src.evals.sts import STSEvaluator
from src.models import GaussOutput
from src.similarity import asymmetrical_kl_sim, symmetrical_kl_sim


class Evaluator:
    def __init__(
        self,
        dataset_dir: Path,
        symmetrical_sim_fn: Callable[[list[str], list[str]], list[float]],
        vec_sim_fn: Callable[[list[str], list[str]], list[float]],
        asymmetrical_sim_fn: Callable[[list[str], list[str]], list[float]],
        dist_fn: Callable[[list[str], list[str]], GaussOutput],
    ) -> None:
        self.dataset_dir = Path(dataset_dir)

        self.nli = NLIEvaluator(dataset_dir, asymmetrical_sim_fn)
        self.sts_vec = STSEvaluator(dataset_dir, vec_sim_fn)
        self.sts = STSEvaluator(dataset_dir, symmetrical_sim_fn)
        self.direction = DirectionPredictionEvaluator(dataset_dir, dist_fn)

    @torch.inference_mode()
    def eval(self):
        return {
            "nli": self.nli.eval(),
            "sts-vec": self.sts_vec.eval(),
            "sts": self.sts.eval(),
            "direction": self.direction.eval(),
        }

    @torch.inference_mode()
    def dev(self) -> float:
        return self.nli.dev_auc()

    @classmethod
    def for_vector(self, encode_fn) -> None:
        raise NotImplementedError

    @classmethod
    def for_gaussian(
        cls,
        dataset_dir: Path,
        encode_fn: Callable[[list[str], list[str]], GaussOutput],
    ) -> None:
        def symmetrical_sim_fn(sent0: list[str], sent1: list[str]) -> list[float]:
            sent0: GaussOutput = encode_fn(sent0)
            sent1: GaussOutput = encode_fn(sent1)
            return symmetrical_kl_sim(sent0.mu, sent0.std, sent1.mu, sent1.std).tolist()

        def vec_sim_fn(sent0: list[str], sent1: list[str]) -> list[float]:
            sent0: GaussOutput = encode_fn(sent0)
            sent1: GaussOutput = encode_fn(sent1)
            cosine_scores = 1 - paired_cosine_distances(
                sent0.mu.float().cpu(),
                sent1.mu.float().cpu(),
            )
            return cosine_scores.tolist()

        def asymmetrical_sim_fn(sent0: list[str], sent1: list[str]) -> list[float]:
            sent0: GaussOutput = encode_fn(sent0)
            sent1: GaussOutput = encode_fn(sent1)
            return asymmetrical_kl_sim(
                sent0.mu, sent0.std, sent1.mu, sent1.std
            ).tolist()

        dist_fn = encode_fn

        return cls(
            dataset_dir, symmetrical_sim_fn, vec_sim_fn, asymmetrical_sim_fn, dist_fn
        )
