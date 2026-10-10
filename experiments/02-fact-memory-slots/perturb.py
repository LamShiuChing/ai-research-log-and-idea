"""Slot-deletion sensitivity -- an R-side proxy measurable before the write path exists.

For a fact the model answers correctly, find the slot its lookup actually selects, zero that
slot's values, and separate two effects:

- **target** -- does that fact stop being answered? Retraction that leaves the fact intact is
  no retraction at all.
- **collateral** -- how much accuracy is lost on *other* facts. Exact retraction means zero.

THEORY.md §7.6 predicts collateral cannot be exactly zero even at top_k = 1, because the read
weights are a softmax whose denominator runs over every slot. Whatever floor appears here is
the gap between "exactly deletable by construction" and what the architecture delivers, and it
is the first R number the project can actually quote.
"""

import json
import sys
from pathlib import Path
from random import Random

import torch

from glt.corpus import build, prefix
from glt.model import GPT, GPTConfig
from glt.train import TrainConfig, accuracy
from glt.universe import generate

CFG = TrainConfig()
# The only non-tensor in our checkpoints is our own config dataclass; allowlisting it
# lets these load under weights_only=True instead of unpickling arbitrary objects.
torch.serialization.add_safe_globals([GPTConfig])


def load(checkpoint: Path) -> GPT:
    ck = torch.load(checkpoint, map_location=CFG.device, weights_only=True)
    model = GPT(ck["config"]).to(CFG.device)
    model.load_state_dict(ck["model"])
    model.requires_grad_(False)  # inference only, and slot rows are edited in place below
    return model.eval()


@torch.no_grad()
def selected_slot(model: GPT, corpus, fact) -> int:
    """The slot this fact's lookup weights most heavily at the answer position."""
    block = next(b for b in model.transformer.h if b.slots is not None)
    reads = {}
    handle = block.slots.register_forward_pre_hook(lambda _m, args: reads.setdefault("x", args[0]))
    model(torch.tensor([corpus.tokenizer.encode(prefix(fact))], device=CFG.device))
    handle.remove()
    return (reads["x"][0, -1] @ block.slots.keys.T).argmax().item()


@torch.no_grad()
def answers(model: GPT, corpus, fact) -> bool:
    ids = torch.tensor([corpus.tokenizer.encode(prefix(fact))], device=CFG.device)
    return model(ids)[0][0, -1].argmax().item() == corpus.tokenizer.stoi[fact.object]


def online_facts(seed: int = 0, n_online: int = 200):
    """Reproduce online.py's held-out split -- same seed, same sample, same facts."""
    universe = generate(seed, entities=200, relations=20, out_degree=10)
    return tuple(Random(seed).sample(universe.base, n_online))


def measure(checkpoint: Path, corpus, n_facts: int = 12, pool=None) -> dict:
    model = load(checkpoint)
    block = next(b for b in model.transformer.h if b.slots is not None)
    pool = corpus.heldout["test_inferred_id"] if pool is None else pool
    baseline = accuracy(model, corpus, pool, CFG)

    sampled = Random(0).sample(pool, min(200, len(pool)))
    correct = [f for f in sampled if answers(model, corpus, f)][:n_facts]
    broken, collateral = 0, []
    for fact in correct:
        slot = selected_slot(model, corpus, fact)
        saved = block.slots.values[slot].clone()
        block.slots.values[slot] = 0
        broken += not answers(model, corpus, fact)
        collateral.append(baseline - accuracy(model, corpus, pool, CFG))
        block.slots.values[slot] = saved

    ck_cfg = model.config
    return {
        "arm": checkpoint.stem,
        "top_k": ck_cfg.slot_top_k,
        "accuracy": round(baseline, 4),
        "tested": len(correct),
        "target_broken": round(broken / max(len(correct), 1), 3),
        "mean_collateral": round(sum(collateral) / max(len(collateral), 1), 5),
        "max_collateral": round(max(collateral, default=0.0), 5),
    }


def main(*names: str) -> None:
    corpus = build(generate(0, entities=200, relations=20, out_degree=10))
    for name in names or ("slots_k1",):
        path = Path("out") / f"{name}.pt"
        if not path.exists():
            print(f"{name}: no checkpoint")
            continue
        # For the online model the facts that live in slots are the held-out ones, so those
        # are the pool a deletion should be measured against.
        pool = online_facts() if name.startswith("online") else None
        print(json.dumps(measure(path, corpus, pool=pool)), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
