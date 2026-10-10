"""Two-digit compositional stream (notes/design-composition.md; HOLES E27): a pair canvas of two
MNIST images (input 1568). Phase 1 primitives: predict both digits (20 logits = two 10-blocks).
Phase 2 composition: binary predicates over the pair — rules on top of fixed inputs, so input
marginals carry no task information."""
import torch

from . import data

DIM = 2 * data.DIM
PREDICATES = dict(parity=lambda l, r: (l + r) % 2,
                  lt=lambda l, r: (l < r).long(),
                  ge10=lambda l, r: (l + r >= 10).long())


def batches(x, y, n_examples, batch, gen):
    """Online pair stream: each canvas is two independently drawn images, each canvas seen once."""
    for i in range(0, n_examples, batch):
        idx = torch.randint(len(x), (min(batch, n_examples - i), 2), generator=gen).to(x.device)
        yield torch.cat([x[idx[:, 0]], x[idx[:, 1]]], 1), y[idx[:, 0]], y[idx[:, 1]]
