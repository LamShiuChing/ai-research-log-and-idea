"""Disjoint memory slots -- the read path (P2).

A trainable key-value store read natively inside the forward pass: gradients flow, no
retrieval call, no external service. Lineage: *Large Memory Layers with Product Keys*
(arXiv:1907.05242) and *Memory Layers at Scale* (arXiv:2412.09764).

`top_k` is THEORY.md §6's sharing knob: 1 is maximally disjoint, `n_slots` is a dense adapter.

`read` decides whether the memory can stay silent, which turns out to matter more:

- **softmax** -- weights are *relative* and sum to 1, so some slot always fires at full
  strength even when nothing relevant is stored. Measured overwriting correct backbone
  answers: facts the memory never held fell 0.996 -> 0.243, with a mean read weight of 0.795
  on queries it had nothing for.
- **gated** -- sigmoid of the score plus one global bias. Fails the same way: the bias learned
  down to -7.17 while the keys simply grew until the sigmoid saturated anyway (gate 0.999,
  zero silent queries out of 128).
- **threshold** -- relu of the score plus a *per-slot* bias, which is what a transformer's own
  feed-forward layer does. Geva et al. (EMNLP 2021, arXiv:2012.14913) show `W_up` rows are keys
  and `W_down` columns are values, so the MLP is already a key-value memory -- and its
  nonlinearity gives abstention for free: an unmatched key goes negative, relu makes it exactly
  zero, and that slot contributes nothing. No gate to learn, no signal needed to learn it.
  It also removes the shared softmax denominator, and with it THEORY.md §7.6's sharing channel.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class MemorySlots(nn.Module):
    """Sparse key-value read. Adds parameters without adding depth."""

    def __init__(self, n_embd: int, n_slots: int, top_k: int, read: str = "softmax"):
        super().__init__()
        self.top_k = min(top_k, n_slots)
        self.read = read
        self.keys = nn.Parameter(torch.randn(n_slots, n_embd) * 0.02)
        # Non-zero on purpose: with values at zero the gradient to the keys vanishes
        # (it flows through the values), so the keys would never move off their init.
        self.values = nn.Parameter(torch.randn(n_slots, n_embd) * 0.02)
        # Per-slot, not global: a single shared bias cannot distinguish a query this slot
        # should answer from one it should ignore, and gradient just opens it for everything.
        # Initialised at 0, matching the backbone's own convention (nanoGPT runs bias=False, so
        # its MLP threshold is exactly 0). A negative init is silent but *dead*: relu zeroes
        # every slot, so no gradient reaches the keys and nothing ever learns to fire.
        self.bias = nn.Parameter(torch.zeros(n_slots)) if read in ("gated", "threshold") else None

    def weights(self, x):
        scores = x @ self.keys.T
        if self.read == "threshold":
            return F.relu(scores + self.bias)
        if self.read == "gated":
            return (scores + self.bias).sigmoid()
        return scores.softmax(-1)

    def lookup(self, addr):
        """Exact addressing: read the slot the address names, or nothing where it is -1.

        Content addressing on hidden states collapses -- 2000 facts routed to 60 of 2048
        slots and 98.7% of old facts landed on a slot a new fact owned, because queries for
        different facts sit at cosine 0.88 and no linear projection recovers the difference.
        An address derived from token identity has no such problem: distinct facts are
        distinct words, so distinct slots, by construction rather than by optimisation.
        """
        return torch.where(addr[..., None] >= 0, self.values[addr.clamp(min=0)], 0.0)

    def forward(self, x, addr=None):
        if self.read == "token":
            return self.lookup(addr)
        weights = self.weights(x)
        if self.top_k < weights.shape[-1]:
            top, idx = weights.topk(self.top_k, dim=-1)
            weights = torch.zeros_like(weights).scatter_(-1, idx, top)
        # Zero-then-matmul rather than gathering values[idx]: the gather materialises
        # (batch, time, top_k, n_embd), which peaked at 1.7 GB for top_k=64 and 3.5 GB dense,
        # doubled by autograd. This form is (batch, time, n_slots) at every k.
        return weights @ self.values
