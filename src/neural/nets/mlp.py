import torch
from torch import nn


class MLP(nn.Module):
    """Fully connected network: n_in -> hidden[0] -> ... -> 1."""

    def __init__(self, n_in: int, hidden: tuple[int, ...] = (64, 64)):
        super().__init__()
        layers, width = [], n_in
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        self.layers = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(1)
