import torch
from torch import nn


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
