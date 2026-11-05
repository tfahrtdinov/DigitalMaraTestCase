import torch
from torch import nn


class LSTM(nn.Module):
    """Encoder over the past hours, decoder over the forecast hours. The encoder's last
    state is appended to the decoder input at every hour; the decoder runs in both
    directions because the inputs of all forecast hours are known at issue."""

    def __init__(
        self,
        n_past: int,
        n_future: int,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        between_layers = dropout if layers > 1 else 0.0
        self.encoder = nn.LSTM(
            n_past, hidden, layers, batch_first=True, dropout=between_layers
        )
        self.decoder = nn.LSTM(
            n_future + hidden,
            hidden,
            layers,
            batch_first=True,
            dropout=between_layers,
            bidirectional=True,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 1))

    def forward(self, past: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """(batch, past hours, n_past), (batch, hours, n_future) -> (batch, hours)."""
        _, (state, _) = self.encoder(past)
        context = state[-1].unsqueeze(1).expand(-1, future.size(1), -1)
        out, _ = self.decoder(torch.cat([future, context], dim=-1))
        return self.head(out).squeeze(-1)
