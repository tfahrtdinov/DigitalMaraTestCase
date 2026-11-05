import torch
from torch import nn
from torch.nn import functional as F


class RegressionModel:
    """L1 loss, Adam."""

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
