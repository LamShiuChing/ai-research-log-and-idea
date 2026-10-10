"""All-layer low-rank adapter with per-layer unit gates (notes/design-composition.md, A1/A5):
every weight gets W0 x + B A x with the base frozen; hidden activations are scaled by gates
(init 1). B = 0 at init, so the composite starts at exactly the base's behaviour and deleting
the adapter restores it bit-for-bit. Gradients flow through the frozen base by construction."""
import torch
from torch import nn


class Adapter(nn.Module):
    """Composite model over a frozen MLP base. forward mirrors MLP.forward (logits, feats)."""

    def __init__(self, base, rank=4):
        super().__init__()
        self.base = [base]  # list, not submodule: the base is not part of the adapter's state
        layers = [*base.hidden, base.out]
        self.A = nn.ParameterList(nn.Parameter(torch.randn(rank, l.in_features) / l.in_features ** 0.5)
                                  for l in layers)
        self.B = nn.ParameterList(nn.Parameter(torch.zeros(l.out_features, rank)) for l in layers)
        self.gate = nn.ParameterList(nn.Parameter(torch.ones(l.out_features)) for l in base.hidden)
        self.to(base.out.weight.device)

    def forward(self, x):
        base = self.base[0]
        feats = []
        for layer, A, B, g in zip(base.hidden, self.A, self.B, self.gate):
            x = torch.relu(layer(x) + x @ A.T @ B.T) * g
            feats.append(x)
        return base.out(x) + x @ self.A[-1].T @ self.B[-1].T, feats

    @torch.no_grad()
    def fold(self):
        """Weight-addition merge (C15d): make the base compute exactly the composite. Deltas add
        into each weight; gates fold into the next layer's input columns. No rehearsal anywhere —
        this is T2 Killer A in its strongest form (the base adopts the composite wholesale)."""
        base = self.base[0]
        for l, A, B in zip([*base.hidden, base.out], self.A, self.B):
            l.weight += B @ A
        for g, nxt in zip(self.gate, [*base.hidden[1:], base.out]):
            nxt.weight *= g

    def delta_norm(self):
        return sum((B @ A).norm() ** 2 for A, B in zip(self.A, self.B)).sqrt().item()
