"""What is worth writing into memory -- an MDL admission policy.

PROJECT_GOAL.md §6 lists **surprise** as a routing signal rather than a truth signal: low
surprise means the fact is already known, so skip the write. Minimum Description Length turns
that from a hand-tuned threshold into an accounting. Cross-entropy *is* bits, so a statement's
loss is exactly what that fact costs to encode under the current model, and the write is worth
making only when it buys back more than it costs.

The honest caveat on the accounting: a slot here is 2*n_embd + 1 parameters, ~8k bits at bf16,
while a fact in this universe carries at most log2(200) = 7.6 bits. Taken literally, MDL says
never store anything -- the slot is enormous relative to its contents. What survives literally
is the *comparison between candidates* and the zero-benefit case: a fact the model already
predicts buys nothing at all, so writing it is pure interference cost. That is the rule
implemented here, and it is the one §6 actually asks for.
"""

import math

import torch

from .corpus import prefix


@torch.no_grad()
def surprise_bits(model, corpus, facts, device: str = "cuda") -> torch.Tensor:
    """Bits needed to encode each fact's answer under the current model: -log2 p(answer)."""
    ids = torch.tensor([corpus.tokenizer.encode(prefix(f)) for f in facts], device=device)
    gold = torch.tensor([corpus.tokenizer.stoi[f.object] for f in facts], device=device)
    logits, _ = model(ids)
    logprobs = logits[:, -1, :].float().log_softmax(-1)
    return -logprobs[range(len(facts)), gold] / math.log(2)


def admit(model, corpus, facts, device: str = "cuda", min_bits: float = 1.0):
    """Facts the model cannot already predict cheaply.

    The default of 1 bit is the break-even reading of the rule: the model assigns the correct
    answer less than even odds, so it does not know the fact and the write can buy something.
    """
    bits = surprise_bits(model, corpus, facts, device)
    return tuple(f for f, b in zip(facts, bits.tolist()) if b >= min_bits)
