"""Training with validation-based model selection and a single test evaluation."""

import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import StandardScaler, WindowDataset
from .metrics import Metrics, masked_mae, valid_mask
from .models import MODELS


def set_seed(seed, threads=1):
    """Seed all generators and make CPU/GPU kernels deterministic."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def resolve_device(name):
    """'auto' picks cuda:0 when available, otherwise the CPU."""
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


class Trainer:
    """Trains one backbone on a set of training-sample starts."""

    def __init__(self, model_name, model_params, training, dataset, H, P, device="auto"):
        """Seed, fit the scaler on the full training split and build the model."""
        self.training = training
        self.dataset = dataset
        self.H, self.P = H, P
        set_seed(training["seed"], training["threads"])
        self.device = resolve_device(device)
        self.scaler = StandardScaler.for_dataset(dataset)
        self.backbone = MODELS[model_name]
        self.model = self.backbone.build(model_params, dataset, H, P).to(self.device)

    def loader(self, split, starts, shuffle=False, epoch=0):
        """DataLoader over the samples at the given starts."""
        windows = WindowDataset(
            self.dataset.values(split),
            self.dataset.timestamps(split),
            starts,
            self.H,
            self.P,
            self.scaler,
        )
        generator = None
        if self.training["loader_seed_mode"] == "epoch":
            generator = torch.Generator().manual_seed(self.training["seed"] * 1000003 + epoch)
        return DataLoader(
            windows,
            batch_size=self.training["batch_size"],
            shuffle=shuffle,
            num_workers=0,
            drop_last=False,
            generator=generator,
        )

    def batches(self, loader):
        """Move batches to the device and add the valid-target mask."""
        for x, y, calendar in loader:
            x, y, calendar = (v.to(self.device) for v in (x, y, calendar))
            yield x, y, calendar, valid_mask(y, self.dataset.missing)

    def train_epoch(self, loader, optimizer):
        """One pass over the training samples; returns the mean loss."""
        self.model.train()
        losses = []
        for x, y, calendar, mask in self.batches(loader):
            optimizer.zero_grad(set_to_none=True)
            prediction = self.backbone.predict(self.model, x, calendar)
            if self.training["loss_space"] == "standardized":
                y = self.scaler.normalize(y)
            else:
                prediction = self.scaler.inverse(prediction)
            loss = masked_mae(prediction, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training["grad_clip"])
            optimizer.step()
            losses.append(loss.item())
        return float(np.mean(losses))

    def evaluate(self, loader, space="original"):
        """Metrics over a whole split, in the original or standardized scale."""
        self.model.eval()
        metrics = Metrics(self.dataset.missing)
        with torch.inference_mode():
            for x, y, calendar, mask in self.batches(loader):
                prediction = self.backbone.predict(self.model, x, calendar)
                if space == "standardized":
                    y = self.scaler.normalize(y)
                else:
                    prediction = self.scaler.inverse(prediction)
                metrics.update(prediction, y, mask)
        return metrics.result()

    def fit(self, starts, output):
        """Train for a fixed number of epochs, keep the best validation epoch and test it once."""
        cfg = self.training
        optimizer_class = {"Adam": torch.optim.Adam, "AdamW": torch.optim.AdamW}[cfg["optimizer"]]
        optimizer = optimizer_class(
            self.model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg["milestones"], gamma=cfg["gamma"]
        )

        H, P = self.H, self.P
        train_loader = self.loader("train", starts, shuffle=True)
        val_loader = self.loader("val", self.dataset.all_starts("val", H, P))
        best_mae, best_epoch = float("inf"), None
        history = output / "history.jsonl"
        history.unlink(missing_ok=True)
        began = time.perf_counter()

        for epoch in range(1, cfg["epochs"] + 1):
            if cfg["loader_seed_mode"] == "epoch":
                train_loader = self.loader("train", starts, shuffle=True, epoch=epoch)
            train_loss = self.train_epoch(train_loader, optimizer)
            val = self.evaluate(val_loader, cfg["validation_space"])
            scheduler.step()
            if val["MAE"] < best_mae:
                best_mae, best_epoch = val["MAE"], epoch
                torch.save(self.model.state_dict(), output / "best.pt")
            with history.open("a", encoding="utf8") as stream:
                stream.write(
                    json.dumps({"epoch": epoch, "train_loss": train_loss, "val": val}) + "\n"
                )
            print(
                f"Epoch {epoch}/{cfg['epochs']}  train loss {train_loss:.4f}  val MAE {val['MAE']:.4f}",
                flush=True,
            )

        self.model.load_state_dict(torch.load(output / "best.pt", map_location=self.device))
        test = self.evaluate(self.loader("test", self.dataset.all_starts("test", H, P)))
        return {
            "best_epoch": best_epoch,
            "best_val_MAE": best_mae,
            "test": test,
            "train_seconds": time.perf_counter() - began,
        }
