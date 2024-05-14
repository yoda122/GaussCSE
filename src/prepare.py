from pathlib import Path

import pandas as pd
from tap import Tap

import src.utils as utils


class Args(Tap):
    dataset_dir: Path = "./datasets"
    seed = 42


def main(args: Args):
    utils.set_seed(args.seed)

    for src, dst in [
        ("train", "train"),
        ("dev", "val"),
        ("test", "test"),
    ]:
        df = utils.load_jsonl(
            args.dataset_dir / f"snli/raw/snli_1.0/snli_1.0_{src}.jsonl"
        )
        df = df[["sentence1", "sentence2", "gold_label"]]
        df.columns = ["premise", "hypothesis", "label"]
        df = df.sample(frac=1).reset_index(drop=True)
        utils.save_jsonl(df, args.dataset_dir / f"snli/{dst}.jsonl")

    for src, dst in [
        ("train", "train"),
        ("dev_matched", "val"),
        ("dev_mismatched", "test"),
    ]:
        df = utils.load_jsonl(
            args.dataset_dir / f"mnli/raw/multinli_1.0/multinli_1.0_{src}.jsonl"
        )
        df = df[["sentence1", "sentence2", "gold_label"]]
        df.columns = ["premise", "hypothesis", "label"]
        df = df.sample(frac=1).reset_index(drop=True)
        utils.save_jsonl(df, args.dataset_dir / f"mnli/{dst}.jsonl")

    df: pd.DataFrame = pd.read_table(
        args.dataset_dir / "sts/sick/SICK_test_annotated.txt"
    )
    df = df[["sentence_A", "sentence_B", "entailment_judgment"]]
    df.columns = ["premise", "hypothesis", "label"]
    df["label"] = df["label"].str.lower()
    df = df.sample(frac=1).reset_index(drop=True)
    utils.save_jsonl(df, args.dataset_dir / "sick/test.jsonl")


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
