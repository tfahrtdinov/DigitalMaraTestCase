import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader

from loggers import Logger
from neural.models import NeuralModel


def _on_device(model: NeuralModel, batch: tuple[torch.Tensor, ...]) -> tuple:
    device = next(model.net.parameters()).device
    return tuple(t.to(device) for t in batch)


class Trainer:
    """Fixed number of epochs. Train loss, and validation loss when `val` is given,
    go to `logger` every epoch. `one_cycle` anneals the learning rate from the optimizer's
    value as the peak; `grad_clip` limits the gradient norm."""

    def __init__(
        self,
        epochs: int = 30,
        threads: int = 1,
        seed: int = 0,
        logger: Logger | None = None,
        device: str = "cpu",
        grad_clip: float | None = None,
        one_cycle: bool = False,
    ):
        self.epochs, self.threads, self.seed, self.logger = (
            epochs,
            threads,
            seed,
            logger,
        )
        self.device, self.grad_clip, self.one_cycle = device, grad_clip, one_cycle

    def fit(
        self, model: NeuralModel, train: DataLoader, val: DataLoader | None = None
    ) -> NeuralModel:
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        model.net.to(self.device)
        optimizer = model.optimizer()
        scheduler = None
        if self.one_cycle:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[g["lr"] for g in optimizer.param_groups],
                total_steps=self.epochs * len(train),
            )
        for epoch in range(self.epochs):
            record = {
                "train_loss": self._train_epoch(model, train, optimizer, scheduler)
            }
            if val is not None:
                record["val_loss"] = self.validate(model, val)
            if self.logger:
                self.logger.log_metrics(record, step=epoch)
        return model

    def _train_epoch(
        self,
        model: NeuralModel,
        train: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ) -> float:
        model.net.train()
        total = 0.0
        for batch in train:
            optimizer.zero_grad()
            loss = model.loss(_on_device(model, batch))
            loss.backward()
            if self.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.net.parameters(), self.grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += loss.item()
        return total / len(train)

    @staticmethod
    @torch.no_grad()
    def validate(model: NeuralModel, val: DataLoader) -> float:
        model.net.eval()
        losses = [model.loss(_on_device(model, batch)).item() for batch in val]
        return float(np.mean(losses))

    @staticmethod
    @torch.no_grad()
    def predict(model: NeuralModel, data: DataLoader) -> NDArray[np.float64]:
        model.net.eval()
        parts = [model.predict(_on_device(model, batch)).cpu() for batch in data]
        return torch.cat(parts).numpy()
