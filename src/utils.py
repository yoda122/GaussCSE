import json
import random
from collections.abc import Iterable
from inspect import ismethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def save_jsonl(data: Iterable | pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if type(data) != pd.DataFrame:
        data = pd.DataFrame(data)
    data.to_json(
        path,
        orient="records",
        lines=True,
        force_ascii=False,
    )


def save_json(data: dict[Any, Any], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_jsonl(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    return pd.read_json(path, lines=True)


def load_json(path: Path | str) -> dict:
    path = Path(path)
    with path.open() as f:
        data = json.load(f)
    return data


def log(data: dict, path: Path | str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df: pd.DataFrame = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        df.to_csv(path, index=False)
    else:
        pd.DataFrame([data]).to_csv(path, index=False)


def save_config(data, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data: dict = data.as_dict()
    except:
        try:
            data: dict = data.to_dict()
        except:
            pass

    if type(data) != dict:
        data: dict = vars(data)

    data = {k: v for k, v in data.items() if not ismethod(v)}
    data = {
        k: v if type(v) in [int, float, bool, None] else str(v) for k, v in data.items()
    }

    save_json(data, path)


def save_csv(
    df: Iterable | pd.DataFrame,
    path: str | Path,
    filter_column: str | list[str] = None,
    filter_value: str | list[str] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if type(df) != pd.DataFrame:
        df = pd.DataFrame(df)

    if filter_column and filter_value:
        df = df[df[filter_column] == filter_value]

    df.to_csv(path, index=False)


def set_seed(seed: int = None) -> None:
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dict_average(dicts: Iterable[dict]) -> dict:
    dicts: list[dict] = list(dicts)
    averaged = {}

    for k, v in dicts[0].items():
        try:
            v = v.item()
        except:
            pass
        if type(v) in [int, float]:
            averaged[k] = v / len(dicts)
        else:
            averaged[k] = [v]

    for d in dicts[1:]:
        for k, v in d.items():
            try:
                v = v.item()
            except:
                pass
            if type(v) in [int, float]:
                averaged[k] += v / len(dicts)
            else:
                averaged[k].append(v)

    return averaged


def torch_dtype(dtype: str) -> torch.dtype:
    match dtype.lower():
        case "bf16":
            return torch.bfloat16
        case "fp16":
            return torch.float16
        case _:
            return torch.float32


def log(data: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        df: pd.DataFrame = pd.read_csv(path)
        df = pd.DataFrame(df.to_dict("records") + [data])
        df.to_csv(path, index=False)
    else:
        pd.DataFrame([data]).to_csv(path, index=False)
