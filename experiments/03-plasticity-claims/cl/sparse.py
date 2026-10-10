"""SLEEP-sparse (CLAIMS SS-a/b/c; the user's synaptic-tagging proposal, 2026-09-02). The wake
adapter's delta B·A IS the tag: sleep negotiates a sparse delta over the top-k |ΔW| positions
of every adapted Linear while the base stays frozen. The teacher is the same frozen base with
the LoRA branch switched on (exactly the composite the full sleep deep-copies — no copy, no
cache); the student is the base plus the sparse delta; the fold at the end is exact and legal
(the negotiation already happened). Two-tier: tag positions whose sign agrees with the base
weight fold as-is (reinforcement), the sign-disagreeing rest are the trained set.
Teacherless (SS-d follow-up): the tag is |dL/dW| of a backward pass over the wake chunk and the
sparse delta trains on the chunk's tokens directly with rehearsal - online sparse GD, no adapter,
no copy (from_grads)."""
import torch
import torch.nn.functional as F
from torch import nn

from cl.lm_adapter import _get, _set, spots


class SparseLinear(nn.Module):
    def __init__(self, base, idx, fixed, n_fixed, teacher=None):
        super().__init__()
        self.frozen, self.teacher = [base], teacher  # list: the base is not this module's state
        self.idx, self.fixed, self.n_fixed = idx, fixed, n_fixed
        self.val = nn.Parameter(torch.zeros(len(idx) - n_fixed, device=idx.device))
        self.teach = False

    def delta(self):
        w = self.frozen[0].weight
        return torch.zeros_like(w).view(-1).index_add(0, self.idx, torch.cat([self.fixed, self.val])
                                                        .to(w.dtype)).view_as(w)

    def forward(self, x):
        if self.teach:
            return self.teacher(x)
        return self.frozen[0](x) + F.linear(x, self.delta())


@torch.no_grad()
def _tag(base, dw, frac, mode):
    """(idx, fixed values, n_fixed) over flat weight `base` from flat score `dw`."""
    n = int(frac * dw.numel())
    idx = torch.randperm(dw.numel(), device=dw.device)[:n] if mode == "random" else dw.abs().topk(n).indices
    aligned = (dw[idx].sign() == base[idx].sign()) if mode == "twotier" else torch.zeros_like(idx, dtype=torch.bool)
    idx = torch.cat([idx[aligned], idx[~aligned]])
    return idx, dw[idx[:aligned.sum()]], int(aligned.sum())


class SparseNursery:
    """Replace adapted Linears in place by sparse-delta wraps; `spots_` = (parent, key, base Linear,
    dense score over base.weight, teacher module or None) per Linear."""

    def __init__(self, spots_, frac, mode="topk"):
        self.wraps = []
        for p, k, base, dw, teacher in spots_:
            idx, fixed, n_fixed = _tag(base.weight.detach().view(-1), dw.view(-1), frac, mode)
            s = SparseLinear(base, idx, fixed, n_fixed, teacher)
            _set(p, k, s)
            self.wraps.append((p, k, s))

    @classmethod
    def from_nursery(cls, nursery, frac, mode="topk"):
        """Tag = the wake adapter's delta B·A; teacher = the composite (the LoRA branch on)."""
        return cls([(p, k, w.frozen[0], (w.B @ w.A).detach(), w) for p, k, w in nursery.wraps], frac, mode)

    @classmethod
    def from_grads(cls, net, frac):
        """Teacherless: tag = |dL/dW| left on the adapted Linears by a backward over the wake chunk."""
        return cls([(p, k, _get(p, k), _get(p, k).weight.grad, None) for p, k in spots(net)], frac)

    def params(self):
        return [s.val for _, _, s in self.wraps]

    def teach(self, on):
        for _, _, s in self.wraps:
            s.teach = on

    def counts(self):
        return dict(train=sum(len(s.val) for _, _, s in self.wraps),
                    fixed=sum(s.n_fixed for _, _, s in self.wraps))

    @torch.no_grad()
    def fold(self):
        """Exact fold of the negotiated delta into the frozen base, then restore the Linears."""
        for p, k, s in self.wraps:
            base = s.frozen[0]
            base.weight.add_(s.delta())
            _set(p, k, base)
