from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
import torch.nn.functional as F
from tap import Tap
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer

import src.utils as utils
from src.evals import Evaluator
from src.models import GaussCSEModel, GaussOutput
from src.similarity import KLSimMat


class Args(Tap):
    dataset_dir: Path = "./datasets"
    train_path: Path = "./datasets/sup-simcse/train.csv"

    model_name: str = "bert-base-uncased"

    batch_size: int = 64
    lr: float = 3e-5
    epochs: int = 1
    sim_type = "pos+rev+neg"

    temperature: float = 0.05
    weight_decay: float = 1e-2

    num_warmup_ratio: float = 0.1
    max_seq_len: int = 64
    with_simcse: bool = False

    num_eval_steps: int = 100
    gradient_checkpointing: bool = True

    seed: int = None
    device: str = "cuda:0"
    dtype: utils.torch_dtype = "bf16"

    def process_args(self):
        date, time = datetime.now().strftime("%Y-%m-%d/%H-%M-%S").split("/")

        self.output_dir = self.make_output_dir(
            "outputs",
            self.model_name,
            self.sim_type,
            date,
            time,
        )

    def make_output_dir(self, *args) -> Path:
        args = [str(a).replace("/", "__") for a in args]
        output_dir = Path(*args)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


class Experiment:
    def __init__(self, args: Args):
        self.args = args

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            model_max_length=self.args.max_seq_len,
            use_fast=False,
        )

        self.model = GaussCSEModel(
            model_name=self.args.model_name,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        self.model = self.model.eval().to(self.args.device)

        self.evaluator = Evaluator.for_gaussian(
            dataset_dir=self.args.dataset_dir,
            encode_fn=self.encode_fn,
        )

        self.train_dataset = pd.read_csv(args.train_path).to_dict("records")
        self.train_dataloader = self.create_loader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
        )

        self.optimizer, self.lr_scheduler = self.create_optimizer(
            train_steps_per_epoch=len(self.train_dataloader),
        )

    def create_optimizer(
        self,
        train_steps_per_epoch: int,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        no_decay = {"bias", "LayerNorm.weight"}
        optimizer_grouped_parameters = [
            {
                "params": [
                    param
                    for name, param in self.model.named_parameters()
                    if name not in no_decay
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    param
                    for name, param in self.model.named_parameters()
                    if name in no_decay
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.args.lr)

        num_training_steps = train_steps_per_epoch * self.args.epochs
        num_warmup_steps = int(num_training_steps * self.args.num_warmup_ratio)

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        return optimizer, lr_scheduler

    def tokenize(self, batch: list[str]) -> BatchEncoding:
        return self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.args.max_seq_len,
        )

    def collate_fn(self, data_list: list[dict]) -> BatchEncoding:
        return BatchEncoding(
            {
                "sent0": self.tokenize([d["sent0"] for d in data_list]),
                "sent1": self.tokenize([d["sent1"] for d in data_list]),
                "hard_neg": self.tokenize([d["hard_neg"] for d in data_list]),
            }
        )

    def create_loader(
        self,
        dataset: list[str] | list[dict],
        collate_fn: Callable = None,
        batch_size: int = None,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> DataLoader:
        return DataLoader(
            dataset,
            collate_fn=collate_fn or self.collate_fn,
            batch_size=batch_size or self.args.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            drop_last=drop_last,
        )

    @torch.inference_mode()
    def encode_fn(
        self,
        sentences: list[str],
        batch_size: int = None,
        **_,
    ) -> GaussOutput:
        self.model.eval()
        data_loader = self.create_loader(
            sentences,
            collate_fn=self.tokenize,
            batch_size=batch_size or self.args.batch_size,
            shuffle=False,
        )

        output: list[GaussOutput] = []
        for batch in data_loader:
            with torch.cuda.amp.autocast(dtype=self.args.dtype):
                out = self.model.forward(**batch.to(self.args.device))
            output.append(out)

        output = GaussOutput(
            mu=torch.cat([out.mu for out in output], dim=0),
            std=torch.cat([out.std for out in output], dim=0),
        )

        return output

    def clone_state_dict(self) -> dict:
        return {k: v.detach().clone().cpu() for k, v in self.model.state_dict().items()}

    def log(self, metrics: dict) -> None:
        utils.log(metrics, self.args.output_dir / "log.csv")
        tqdm.write(
            f"epoch: {metrics['epoch']} \t"
            f"step: {metrics['step']} \t"
            f"loss: {metrics['loss']:2.6f}       \t"
            f"dev-auc: {metrics['dev-auc']:.4f}"
        )


def main(args: Args):
    utils.set_seed(args.seed)
    exp = Experiment(args)

    best_dev_score = exp.evaluator.dev()
    best_epoch, best_step = 0, 0
    val_metrics = {
        "epoch": best_epoch,
        "step": best_step,
        "loss": float("inf"),
        "dev-auc": best_dev_score,
    }
    exp.log(val_metrics)
    best_state_dict = exp.clone_state_dict()

    scaler = torch.cuda.amp.GradScaler()
    current_step = 0
    sim_mat_fn = KLSimMat(args.sim_type)

    for epoch in trange(
        args.epochs,
        leave=False,
        dynamic_ncols=True,
        desc="Epoch",
    ):
        train_losses = []
        exp.model.train()

        for batch in tqdm(
            exp.train_dataloader,
            total=len(exp.train_dataloader),
            dynamic_ncols=True,
            leave=False,
            desc="Step",
        ):
            current_step += 1
            batch: BatchEncoding = batch.to(args.device)
            with torch.cuda.amp.autocast(dtype=args.dtype):
                pre: GaussOutput = exp.model.forward(**batch.sent0)
                ent: GaussOutput = exp.model.forward(**batch.sent1)
                con: GaussOutput = exp.model.forward(**batch.hard_neg)

            sim_mat: torch.FloatTensor = sim_mat_fn(pre, ent, con)
            sim_mat: torch.FloatTensor = sim_mat / args.temperature

            batch_size = sim_mat.size(0)
            labels = torch.arange(batch_size).to(args.device, non_blocking=True)
            loss = F.cross_entropy(sim_mat, labels)

            # マルチタスク学習
            if args.with_simcse:
                sim_mat_1st = F.cosine_similarity(
                    pre.mu.unsqueeze(1), ent.mu.unsqueeze(0), dim=-1
                )
                sim_mat_2nd = F.cosine_similarity(
                    pre.mu.unsqueeze(1), con.mu.unsqueeze(0), dim=-1
                )
                sim_mat_simcse = torch.cat([sim_mat_1st, sim_mat_2nd], dim=1)
                sim_mat_simcse = sim_mat_simcse / args.temperature
                loss = loss + F.cross_entropy(sim_mat_simcse, labels)

            train_losses.append(loss.item())

            exp.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(exp.optimizer)

            scale = scaler.get_scale()
            scaler.update()
            if scale <= scaler.get_scale():
                exp.lr_scheduler.step()

            if current_step % args.num_eval_steps == 0:
                exp.model.eval()
                dev_score = exp.evaluator.dev()

                if best_dev_score < dev_score:
                    best_dev_score = dev_score
                    best_epoch, best_step = epoch, current_step
                    best_state_dict = exp.clone_state_dict()

                val_metrics = {
                    "epoch": epoch,
                    "step": current_step,
                    "loss": sum(train_losses) / len(train_losses),
                    "dev-auc": dev_score,
                }
                exp.log(val_metrics)
                train_losses = []

                exp.model.train()

    dev_metrics = {
        "best-epoch": best_epoch,
        "best-step": best_step,
        "best-dev-auc": best_dev_score,
    }
    utils.save_json(dev_metrics, args.output_dir / "dev-metrics.json")

    exp.model.load_state_dict(best_state_dict)
    exp.model.eval().to(args.device)

    metrics = exp.evaluator.eval()
    utils.save_json(metrics, args.output_dir / "metrics.json")
    utils.save_config(args, args.output_dir / "config.json")


if __name__ == "__main__":
    args = Args().parse_args()
    main(args)
