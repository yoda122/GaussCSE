import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from more_itertools import flatten
from tap import Tap


class Args(Tap):
    input_dir: Path = "./outputs"
    output_dir: Path = "./outputs/results"


def extract(metrics_path: Path) -> tuple[str, dict]:
    try:
        dir = metrics_path.parent
        with (dir / "config.json").open() as f:
            config = json.load(f)

        model_name = config.get("model_name")
        batch_size = config.get("batch_size")
        lr = str(config.get("lr"))
        sim_type = str(config.get("sim_type"))

        with metrics_path.open() as f:
            metrics = json.load(f)
        with (dir / "dev-metrics.json").open() as f:
            dev_metrics = json.load(f)

        data = {
            "model_name": model_name,
            "sim_type": sim_type,
            "batch_size": batch_size,
            "lr": lr,
            **dev_metrics,
            **metrics,
        }
        return data

    except Exception as e:
        print(e)
        return None


def main(args: Args):
    dirs = [
        dir
        for dir in args.input_dir.glob("*")
        if not str(dir).startswith("outputs/prev")
    ]
    paths = list(flatten(dir.glob("**/metrics.json") for dir in dirs))

    data = []
    with ThreadPoolExecutor(max_workers=256) as executor:
        for row in executor.map(extract, paths):
            if row is None:
                continue
            data.append(row)

    df = pd.DataFrame(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "all.csv", index=False)


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
