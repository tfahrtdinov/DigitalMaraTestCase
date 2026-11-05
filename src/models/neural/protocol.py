from typing import Protocol

import torch
from torch import nn


class NeuralModel(Protocol):
    """Loss, prediction and optimiser around a network `net`. The Trainer uses nothing else."""

    net: nn.Module

    def loss(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor: ...

    def predict(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor: ...

    def optimizer(self) -> torch.optim.Optimizer: ...
