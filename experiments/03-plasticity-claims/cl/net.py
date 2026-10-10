import torch
from torch import nn


class MLP(nn.Module):
    """ReLU MLP that also returns every hidden activation (needed by CBP utilities and diagnostics)."""

    def __init__(self, d_in=784, hidden=(256, 256, 256), d_out=10):
        super().__init__()
        sizes = (d_in, *hidden)
        self.hidden = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))
        self.out = nn.Linear(sizes[-1], d_out)

    def forward(self, x):
        feats = []
        for layer in self.hidden:
            x = torch.relu(layer(x))
            feats.append(x)
        return self.out(x), feats
