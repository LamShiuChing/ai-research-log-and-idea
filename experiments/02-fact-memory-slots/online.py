"""Can a fact be learned into slots alone, with the backbone frozen? (P3 prototype)

P2 established that *adding* slots does not put facts in them: with the whole model trainable,
gradient descent stores facts in the backbone and zeroing every slot costs 0.0000 atomic
accuracy. Nothing forces the memory tier to carry anything.

This is the corrected experiment, and it is the GLT design rather than an approximation of it:

  phase 1  train the whole model on most of the facts     -> backbone learns structure
  phase 2  FREEZE the backbone, train slots only on facts it has never seen

Phase 2 is "sparse memory finetuning" (arXiv:2605.03229) used as a write path. Because the
backbone cannot move, anything the model learns in phase 2 must be in the slots -- so the
measurements finally mean what they say:

  G2  recall of held-out facts after phase 2
  G5  zero the slot a fact selects: does that fact break (target), and what does it cost
      every other fact (collateral)?
"""

import json
import sys
from pathlib import Path
from random import Random

import numpy as np
import torch

from glt.admission import admit
from glt.corpus import FUNCTION_WORDS, PAD, Tokenizer, build, prefix
from glt.model import GPT, GPTConfig
from glt.questions import cooccurring, statement
from glt.train import TrainConfig, accuracy, get_batch
from glt.universe import absences, generate


def coerce(value: str):
    """CLI values are strings; numeric options must survive as numbers, `read` as text."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def rows(tokenizer: Tokenizer, facts, width: int) -> np.ndarray:
    encoded = [tokenizer.encode(statement(f)) for f in facts]
    out = np.full((len(encoded), width), tokenizer.stoi[PAD], dtype=np.uint16)
    for row, ids in zip(out, encoded):
        row[: len(ids)] = ids
    return out


def freeze_backbone(model: GPT) -> int:
    """Everything but the slot table stops moving. Returns the trainable parameter count."""
    model.requires_grad_(False)
    slots = next(b.slots for b in model.transformer.h if b.slots is not None)
    slots.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def direct_write(model: GPT, corpus, facts, device: str) -> int:
    """Content-addressed write: a slot's key *is* the query vector of the fact it stores.

    Gradient cannot satisfy THEORY.md §4.1 -- it moves whatever reduces the loss, and a write
    phase never sees a query that should be answered with silence, so keys spread to cover
    everything (measured: gate 0.999 on queries the memory has nothing for). Setting the key
    directly puts it exactly where its fact is, and it drifts nowhere because it is never
    trained. Only the matching value row is then learned.
    """
    slots = next(b.slots for b in model.transformer.h if b.slots is not None)
    seen = {}
    handle = slots.register_forward_pre_hook(lambda _m, args: seen.__setitem__("x", args[0]))
    base = slots.keys.shape[0] - len(facts)
    for i, fact in enumerate(facts):
        model(torch.tensor([corpus.tokenizer.encode(prefix(fact))], device=device))
        slots.keys[base + i] = seen["x"][0, -1]
    handle.remove()
    return base


def restrict_writes(model: GPT, n_writable: int) -> int:
    """Confine gradient to the last `n_writable` slot rows.

    THEORY.md §4.1 requires that writing slot k touch only slot k. Letting gradient reach the
    whole table instead reorganises the address space, so previously-learned queries get
    re-routed onto rows that now hold something else -- measured at 76% collateral loss.
    """
    slots = next(b.slots for b in model.transformer.h if b.slots is not None)
    mask = torch.zeros_like(slots.keys[:, :1])
    mask[-n_writable:] = 1.0
    slots.keys.register_hook(lambda g: g * mask)
    slots.values.register_hook(lambda g: g * mask)
    return n_writable * slots.keys.shape[1] * 2


def run(iters_pretrain=60_000, iters_online=20_000, n_online=200, n_slots=2048, top_k=1, seed=0,
        confine=1, read="softmax", direct=0, admission=1, absent=0):
    universe = generate(seed, entities=200, relations=20, out_degree=10)
    corpus = build(universe)
    tokenizer, width = corpus.tokenizer, corpus.train.shape[1]

    # Hold out atomic facts entirely: absent from phase 1, so phase 2 is the only chance to learn them.
    online = Random(seed).sample(universe.base, n_online)
    held = set(online)
    kept = [f for f in universe.base if f not in held]
    kept += [d.fact for d in universe.derived if d.premises.isdisjoint(held)]

    # Questions with no answer, answered by silence. Without them the model has never seen a
    # case where the right output is "nothing", so it emits its best guess for every query it
    # is handed -- measured at 22% confident confabulation. A tenth is held out to score
    # whether silence *generalises* rather than being memorised pair by pair.
    quiet = Random(seed + 1).sample(absences(universe), len(absences(universe))) if absent else []
    cut = round(len(quiet) * 0.9)
    kept += quiet[:cut]
    pretrain_rows = rows(tokenizer, kept, width)

    # Consequences of the held-out facts: absent from phase 1 in every form, and never shown
    # in phase 2 either -- only their premises are. Answering them means an online-learned
    # fact propagated, which is PROJECT_GOAL.md §4's north-star metric.
    adjacent = cooccurring(universe)
    consequences = tuple(d.fact for d in universe.derived
                         if not d.premises.isdisjoint(held)
                         and d.fact.object not in adjacent[d.fact.subject])
    evals = {"online_recall": tuple(online), "online_composition": consequences,
             "kept_atomic": tuple(f for f in universe.base if f not in held)}
    if absent:
        evals["absent"] = tuple(quiet[cut:])
    print("eval sets: " + "  ".join(f"{k}={len(v)}" for k, v in evals.items()), flush=True)

    cfg = TrainConfig(max_iters=iters_pretrain, batch_size=512, n_slots=n_slots, slot_top_k=top_k,
                      slot_read=read)
    torch.manual_seed(seed)
    model = GPT(GPTConfig(vocab_size=len(tokenizer), block_size=width, n_layer=cfg.n_layer,
                          n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=cfg.dropout, bias=cfg.bias,
                          n_slots=n_slots, slot_top_k=top_k, slot_read=read,
                          n_function=len(FUNCTION_WORDS),
                          slot_trigger=tokenizer.stoi['is'])).to(cfg.device)

    # Phase 1 is identical across phase-2 variants, so cache it: iterating on the write path
    # costs 2 minutes instead of 12.
    cache = Path(f"out/phase1_s{n_slots}_k{top_k}_{read}_i{iters_pretrain}{'_abs' if absent else ''}.pt")
    if cache.exists():
        model.load_state_dict(torch.load(cache, map_location=cfg.device, weights_only=True)["model"])
        print(f"phase 1: loaded cached backbone from {cache}", flush=True)
    else:
        print(f"phase 1: {len(pretrain_rows):,} rows, {n_online} atomic facts held out", flush=True)
        optimizer = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate, cfg.betas, cfg.device)
        train_loop(model, optimizer, pretrain_rows, corpus, cfg, iters_pretrain, evals, "pretrain")
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict()}, cache)

    if admission:
        # MDL gate (PROJECT_GOAL.md sec.6): a fact the model already predicts buys no bits, so
        # writing it is pure interference cost.
        candidates, online = online, list(admit(model, corpus, tuple(online), cfg.device))
        print(f"admission: {len(online)}/{len(candidates)} candidates worth writing", flush=True)

    trainable = freeze_backbone(model)
    if direct:
        direct_write(model, corpus, online, cfg.device)
        slots = next(b.slots for b in model.transformer.h if b.slots is not None)
        slots.keys.requires_grad_(False)  # addressing is set by content, never learned
    writable = restrict_writes(model, n_online) if (confine or direct) else trainable
    print(f"\nphase 2: backbone frozen, {trainable:,} slot params, {writable:,} writable"
          f" ({'direct keys, confined values' if direct else 'confined to fresh rows' if confine else 'whole table'})", flush=True)
    online_rows = rows(tokenizer, online, width)
    # No weight decay while writing: AdamW decays every parameter in the group regardless of
    # gradient, which would erode the rows the mask is meant to protect.
    optimizer = model.configure_optimizers(0.0, cfg.learning_rate, cfg.betas, cfg.device)
    train_loop(model, optimizer, online_rows, corpus, cfg, iters_online, evals, "online")
    return model, corpus, evals


def train_loop(model, optimizer, data, corpus, cfg, iters, evals, tag):
    pad = corpus.tokenizer.stoi[PAD]
    for it in range(iters + 1):
        for group in optimizer.param_groups:
            group["lr"] = cfg.learning_rate * min(1.0, (it + 1) / cfg.warmup_iters)
        x, y = get_batch(data, pad, cfg)
        with torch.amp.autocast(cfg.device, dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if it % max(iters // 8, 1) == 0:
            model.eval()
            scores = {name: accuracy(model, corpus, facts, cfg) for name, facts in evals.items()}
            model.train()
            print(f"  {tag} iter={it} loss={loss.item():.3f} "
                  + " ".join(f"{k}={v:.3f}" for k, v in scores.items()), flush=True)


if __name__ == "__main__":
    options = {k: coerce(v) for k, v in (a.split("=", 1) for a in sys.argv[1:])}
    model, corpus, evals = run(**options)
    out = f"out/online{'_abstain' if options.get('absent') else ''}.pt"
    torch.save({"model": model.state_dict(), "config": model.config}, out)
    print("saved", out)
    print(json.dumps({k: len(v) for k, v in evals.items()}))
