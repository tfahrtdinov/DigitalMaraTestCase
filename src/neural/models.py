"""Loss, prediction and optimiser around a network; `Trainer` uses nothing else."""

from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F


class NeuralModel(Protocol):
    """Loss, prediction and optimiser around a network `net`. The Trainer uses nothing else."""

    net: nn.Module

    def loss(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor: ...

    def predict(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor: ...

    def optimizer(self) -> torch.optim.Optimizer: ...


class RegressionModel:
    """L1 loss, Adam. A batch is (x, y)."""

    def __init__(self, net: nn.Module, lr: float = 1e-3, weight_decay: float = 0.0):
        self.net = net
        self.lr, self.weight_decay = lr, weight_decay

    def loss(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        x, y = batch
        return F.l1_loss(self.net(x), y)

    def predict(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.net(batch[0])

    def optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


class RNNModel:
    """L1 loss over the hours selected by a mask, AdamW. A batch is (past, future, y, mask)."""

    def __init__(self, net: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-2):
        self.net = net
        self.lr, self.weight_decay = lr, weight_decay

    def loss(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        past, future, y, mask = batch
        return (torch.abs(self.net(past, future) - y) * mask).sum() / mask.sum()

    def predict(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.net(batch[0], batch[1])

    def optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


MODELS = {"regression": RegressionModel, "rnn": RNNModel}
