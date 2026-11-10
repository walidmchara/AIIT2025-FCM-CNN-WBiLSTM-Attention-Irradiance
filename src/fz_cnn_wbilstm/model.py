"""CNN–Morlet-enhanced BiLSTM–attention regression model.

The Morlet activation operates on recurrent features; cell-gate replacement
is not implied because the publication does not specify such equations.
"""
from __future__ import annotations
import math
import torch
from torch import nn


def morlet(x: torch.Tensor, frequency: float = 1.75) -> torch.Tensor:
    """Real Morlet wavelet cos(1.75 x) exp(-x^2/2), as in the article."""
    return torch.cos(frequency * x) * torch.exp(-0.5 * x.square())


class FuzzyCNNWaveletBiLSTMAttention(nn.Module):
    def __init__(self, input_dim: int, conv_filters=(64, 128, 256),
                 kernel_sizes=(5, 5, 5), pool_sizes=(3, 3, 3),
                 lstm_units: int = 150, attention_units: int = 300,
                 dropout: float = .4, use_morlet: bool = True):
        super().__init__()
        if len(conv_filters) != 3 or len(kernel_sizes) != 3 or len(pool_sizes) != 3:
            raise ValueError("The proposed CNN uses three convolution and pooling blocks")
        self.blocks = nn.ModuleList()
        channels = input_dim
        for filters, kernel, pool in zip(conv_filters, kernel_sizes, pool_sizes):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(channels, filters, kernel_size=kernel, padding=kernel // 2),
                nn.ReLU(), nn.MaxPool1d(pool, stride=pool)))
            channels = filters
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=channels, hidden_size=lstm_units,
                            batch_first=True, bidirectional=True)
        self.use_morlet = use_morlet
        self.attention = nn.Sequential(nn.Linear(2 * lstm_units, attention_units), nn.Tanh(),
                                       nn.Linear(attention_units, 1))
        self.regressor = nn.Linear(2 * lstm_units, 1)
        self._last_attention: torch.Tensor | None = None

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.ndim != 3:
            raise ValueError("Expected [batch, lookback, channels]")
        x = windows.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x.transpose(1, 2))
        x, _ = self.lstm(x)
        if self.use_morlet:
            x = morlet(x)
        scores = self.attention(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        self._last_attention = weights.detach()
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return self.regressor(pooled).squeeze(-1)
