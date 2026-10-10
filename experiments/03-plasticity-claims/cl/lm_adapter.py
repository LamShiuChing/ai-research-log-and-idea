"""LoRA nursery for the tier-2 GPT (C15's adapter ported per design-scale: identity-init
low-rank deltas on every block's qkv/proj/mlp projections + the head; the composite starts at
exactly the base's behaviour and detach() restores the base bit-for-bit). Merge is by
distillation (lm_sleep_run.py); fold() exists only as the weight-addition arm — industry's
LoRA-merge, C15d's predicted-fail re-cut at LM scale (C32c)."""
import torch
from torch import nn


class LoraLinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.frozen = [base]  # list, not submodule: the base is not part of the adapter's state
        self.A = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device)
                              / base.in_features ** 0.5)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    def forward(self, x):
        return self.frozen[0](x) + x @ self.A.T @ self.B.T


def _get(parent, key):
    return parent[key] if isinstance(key, int) else getattr(parent, key)


def _set(parent, key, val):
    if isinstance(key, int):
        parent[key] = val
    else:
        setattr(parent, key, val)


def spots(net):
    """(parent, key) of every adapted Linear: each block's qkv/proj/mlp projections + the head
    (a substrate that names its own, cl/hf.py, is asked instead)."""
    if hasattr(net, "spots"):
        return net.spots()
    out = [(blk, k) for blk in net.blocks for k in ("qkv", "proj")]
    out += [(blk.mlp, i) for blk in net.blocks for i in (0, 2)]
    return out + [(net, "head")]


class Nursery:
    """Attach a fresh identity-init adapter over a GPT in place; detach restores it exactly."""

    def __init__(self, net, rank=8):
        self.wraps = [(p, k, LoraLinear(_get(p, k), rank)) for p, k in spots(net)]
        for p, k, w in self.wraps:
            _set(p, k, w)

    def params(self):
        return [q for _, _, w in self.wraps for q in (w.A, w.B)]

    def detach(self):
        for p, k, w in self.wraps:
            _set(p, k, w.frozen[0])

    @torch.no_grad()
    def fold(self):
        """Weight-addition merge (the industry LoRA-merge; C32c's arm), then detach."""
        for _, _, w in self.wraps:
            w.frozen[0].weight += w.B @ w.A
        self.detach()

    def delta_norm(self):
        return sum((w.B @ w.A).norm() ** 2 for _, _, w in self.wraps).sqrt().item()
