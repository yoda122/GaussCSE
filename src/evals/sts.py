from pathlib import Path
from typing import Callable

import torch
from scipy.stats import spearmanr
from tqdm import tqdm


class STSEvaluatorBase:
    def __init__(
        self,
        sentences1: list[str],
        sentences2: list[str],
        scores: list[float],
    ):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        assert len(self.sentences1) == len(self.sentences2) == len(self.scores)

    def __call__(
        self,
        sim_fn: Callable[[list[str], list[str]], list[float]],
    ) -> float:
        similarities = sim_fn(self.sentences1, self.sentences2)
        spearman = float(spearmanr(self.scores, similarities)[0]) * 100
        return spearman


class SICKEvaluator(STSEvaluatorBase):
    def __init__(self, data_dir: Path):
        sentences1, sentences2, scores = [], [], []

        with (data_dir / "sick/SICK_test_annotated.txt").open() as f:
            _ = next(f)
            for line in f:
                _, sentence1, sentence2, score, *_ = line.strip().split("\t")
                sentences1.append(sentence1)
                sentences2.append(sentence2)
                scores.append(float(score))

        super().__init__(sentences1, sentences2, scores)


class STSBDevEvaluator(STSEvaluatorBase):
    def __init__(self, data_dir: Path):
        sentences1, sentences2, scores = [], [], []
        with (data_dir / "stsb/sts-dev.csv").open() as f:
            for line in f:
                _, _, _, _, score, sentence1, sentence2, *_ = line.strip().split("\t")
                sentences1.append(sentence1)
                sentences2.append(sentence2)
                scores.append(float(score))

        super().__init__(sentences1, sentences2, scores)


class STSBEvaluator(STSEvaluatorBase):
    def __init__(self, data_dir: Path):
        sentences1, sentences2, scores = [], [], []
        with (data_dir / "stsb/sts-test.csv").open() as f:
            for line in f:
                _, _, _, _, score, sentence1, sentence2, *_ = line.strip().split("\t")
                sentences1.append(sentence1)
                sentences2.append(sentence2)
                scores.append(float(score))

        super().__init__(sentences1, sentences2, scores)


class STS16Evaluator(STSEvaluatorBase):
    SUBSETS = [
        "answer-answer",
        "headlines",
        "plagiarism",
        "postediting",
        "question-question",
    ]

    def __init__(self, data_dir: Path):
        sentences1, sentences2, scores = [], [], []

        for subset in self.SUBSETS:
            with (data_dir / f"sts16/STS2016.gs.{subset}.txt").open() as gs, (
                data_dir / f"sts16/STS2016.input.{subset}.txt"
            ).open() as f:
                for line_input, line_gs in zip(f, gs):
                    sentence1, sentence2, *_ = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sentences1.append(sentence1)
                    sentences2.append(sentence2)
                    scores.append(float(line_gs.strip()))

        super().__init__(sentences1, sentences2, scores)


class STS15Evaluator(STSEvaluatorBase):
    SUBSETS = [
        "answers-forums",
        "answers-students",
        "belief",
        "headlines",
        "images",
    ]

    def __init__(self, data_dir: Path):
        sentences1, sentences2, scores = [], [], []

        for subset in self.SUBSETS:
            with (data_dir / f"sts15/STS.gs.{subset}.txt").open() as gs, (
                data_dir / f"sts15/STS.input.{subset}.txt"
            ).open() as f:
                for line_input, line_gs in zip(f, gs):
                    sentence1, sentence2, *_ = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sentences1.append(sentence1)
                    sentences2.append(sentence2)
                    scores.append(float(line_gs.strip()))

        super().__init__(sentences1, sentences2, scores)


class STS14Evaluator(STSEvaluatorBase):
    SUBSETS = [
        "deft-forum",
        "deft-news",
        "headlines",
        "images",
        "OnWN",
        "tweet-news",
    ]

    def __init__(self, data_dir: Path):
        sentences1, sentences2, scores = [], [], []

        for subset in self.SUBSETS:
            with (data_dir / f"sts14/STS.gs.{subset}.txt").open() as gs, (
                data_dir / f"sts14/STS.input.{subset}.txt"
            ).open() as f:
                for line_input, line_gs in zip(f, gs):
                    sentence1, sentence2, *_ = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sentences1.append(sentence1)
                    sentences2.append(sentence2)
                    scores.append(float(line_gs.strip()))

        super().__init__(sentences1, sentences2, scores)


class STS13Evaluator(STSEvaluatorBase):
    SUBSETS = ["FNWN", "headlines", "OnWN"]

    # STS13 here does not contain the "SMT" subtask due to LICENSE issue
    def __init__(self, data_dir: Path):
        sentences1, sentences2, scores = [], [], []

        for subset in self.SUBSETS:
            with (data_dir / f"sts13/STS.gs.{subset}.txt").open() as gs, (
                data_dir / f"sts13/STS.input.{subset}.txt"
            ).open() as f:
                for line_input, line_gs, *_ in zip(f, gs):
                    sentence1, sentence2 = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sentences1.append(sentence1)
                    sentences2.append(sentence2)
                    scores.append(float(line_gs.strip()))

        super().__init__(sentences1, sentences2, scores)


class STS12Evaluator(STSEvaluatorBase):
    SUBSETS = [
        "MSRpar",
        "MSRvid",
        "SMTeuroparl",
        "surprise.OnWN",
        "surprise.SMTnews",
    ]

    def __init__(self, data_dir: Path):
        sentences1, sentences2, scores = [], [], []

        for subset in self.SUBSETS:
            with (data_dir / f"sts12/STS.gs.{subset}.txt").open() as gs, (
                data_dir / f"sts12/STS.input.{subset}.txt"
            ).open() as f:
                for line_input, line_gs in zip(f, gs):
                    sentence1, sentence2, *_ = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sentences1.append(sentence1)
                    sentences2.append(sentence2)
                    scores.append(float(line_gs.strip()))

        super().__init__(sentences1, sentences2, scores)


class STSEvaluator:
    def __init__(
        self,
        dataset_dir: Path,
        sim_fn: Callable[[list[str], list[str]], list[float]],
    ) -> None:
        self.data_dir = dataset_dir / "sts"
        self.sim_fn = sim_fn

        self.sts_evaluators = {
            "sts12": STS12Evaluator(data_dir=self.data_dir),
            "sts13": STS13Evaluator(data_dir=self.data_dir),
            "sts14": STS14Evaluator(data_dir=self.data_dir),
            "sts15": STS15Evaluator(data_dir=self.data_dir),
            "sts16": STS16Evaluator(data_dir=self.data_dir),
            "stsb": STSBEvaluator(data_dir=self.data_dir),
            "sick": SICKEvaluator(data_dir=self.data_dir),
        }
        self.dev_evaluator = STSBDevEvaluator(data_dir=self.data_dir)

    @torch.inference_mode()
    def eval(self) -> dict[str, float]:
        results = {}
        for name, evaluator in tqdm(
            list(self.sts_evaluators.items()),
            dynamic_ncols=True,
            leave=False,
            desc="STS",
        ):
            results[name] = evaluator(self.sim_fn)
        results["avg"] = sum(results.values()) / len(results)
        return results

    @torch.inference_mode()
    def dev(self) -> float:
        return self.dev_evaluator(self.sim_fn)
