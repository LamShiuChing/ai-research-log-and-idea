"""A continual GPT: nanoGPT's code redesigned with the brain's cortex and sleep. Can it learn new kinds of English day
after day without forgetting the old ones?

Why: the brain needs a soft side (word meaning, prediction as the always-on signal). A GPT learns it well but forgets:
every batch nudges every weight toward what it reads now. The brain does not forget because learning touches only the
parts in use, important parts move slowly, and sleep keeps what it knew. Here the same ideas go into a GPT's code.
Base: nanoGPT (github.com/karpathy/nanoGPT, model.py): byte-level (256 symbols, like the brain's rows), 4 blocks,
width 256, 4 heads, 256-byte context; trained from scratch (no pretraining).
Changes (the arms add them one at a time):
  slots    the MLP (c_fc -> gelu -> c_proj) becomes a cortex of 1,024 key/value slots (same size as the MLP): each
           position reads only its 32 best-matching slots, so learning changes only those. Each slot has an importance
           (its share of use on the days it served); a slot's change per step is scaled by 1 / (1 + importance).
           After day 1 the shared parts (attention, word table, norms, head) learn at 0.1x: the slow cortex.
  dreams   sleep: before a night, the model's own copy dreams 512 texts of 256 bytes (started from a newline, no stored
           old text); during the night each step also asks the model to keep its old predictions on a dream (KL to the
           copy). What it knew is kept by rehearsing its own dreams, not by storing the past.
Days (real text, BabyLM 10M "clean", one domain a day; the last 5% of each file is held out):
  CHILDES speech to children, Simple Wikipedia, BNC spoken, Gutenberg, QED subtitles, children's stories.
  A day = 1,500 steps of 32 x 256 bytes on that domain's text (the diary); learning happens at night on the diary.
Each day (the user's protocol): load yesterday's checkpoint (warm start), learn one dataset, save a checkpoint, then
use it: test on every dataset's held-out text and let it write a 160-byte sample.
Arms: plain (nanoGPT, day by day), slots, dreams, slots + dreams; joint (plain, all domains mixed, same total steps:
the ceiling). Measured after every night: bits per byte (lower = better) on every domain's held-out text.
  just learned   a domain's bits right after its own night       forgetting   its bits at the end minus just learned
Not in this test: the tally-sheet head and the hippocampus look-alike recall (one-shot words): next steps.

Predictions, stated before the first run (one seed):
  G1 plain forgets: average forgetting over days 1-5 >= 0.10 bits per byte.
  G2 slots + dreams forgets <= 1/3 of plain's.
  G3 it still learns: its just-learned bits are within 0.05 of plain's on average.
  G4 overall: its final average over the 6 domains is below plain's and within 0.10 of joint's.

Results (results/cgpt.json, results/cgpt.log; one seed, ~8 minutes). Bits per byte, lower = better:
                    just learned   forgetting   final average
  plain                 1.825         0.583         2.311
  slots                 2.293         0.244         2.497
  dreams                1.828         0.205         1.999
  slots + dreams        2.334        -0.003         2.331
  joint (ceiling)                                   1.812
Verdicts (the predictions were about slots + dreams):
- G1 passed: plain forgets 0.58 bits per byte on average (line 0.10); e.g. Simple Wikipedia 2.01 -> 2.73.
- G2 passed: slots + dreams forgets nothing (-0.003; line 0.19).
- G3 failed: it learns each new domain far worse (2.33 vs 1.83; line within 0.05).
- G4 failed: final average 2.33, not below plain's 2.31, and 0.52 above joint (line 0.10).
Post hoc:
- Dreams alone (plain GPT + sleep on its own dreams) is the best arm: learns exactly as well as plain (1.828 vs
  1.825), forgets 0.205 (35% of plain's; just misses the 1/3 line set for the combined arm), final 2.00, 0.19 above
  joint. No stored old text was used.
- Slots cost learning from day 1 (child speech 2.16 vs 1.77, before any slowing applies): 32 of 1,024 slots per
  position is a weaker layer than the dense MLP of the same size; after day 1 the 0.1x shared rate slows it further.
  Their low forgetting is partly learning less. As built, slots do not help.
"""
import copy
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

ROOT = Path(__file__).resolve().parent
DOMAINS = ["aochildes", "simple_wikipedia", "bnc_spoken", "gutenberg", "qed", "children_stories"]
CTX, D, HEADS, LAYERS, SLOTS, TOPK = 256, 256, 4, 4, 1024, 32
STEPS, BATCH, LR, SLOW, DREAMS, DREAM_WEIGHT, TEST_WINDOWS = 1500, 32, 1e-3, 0.1, 512, 1.0, 64
DEV = "cuda"
PAD = VOCAB = 256
GELU, TOKENIZER = "none", None  # bytes by default; GPT-2 word pieces in the gpt2 sizes
STARTS = b"abcdefghijklmnopqrstuvwxyz \n"
CHECKPOINTS = ROOT / "results" / "checkpoints"
SIZES = {"small": (256, 4, 4, 1e-3), "big": (768, 12, 12, 3e-4), "gpt2-mini": (512, 8, 4, 1e-4)}  # width, heads,
# blocks, learning rate


def set_size(name):
    """Model size for every GPT built afterwards (big = GPT-2 small's shape, ~85M numbers)."""
    global D, HEADS, LAYERS, LR, VOCAB, PAD, GELU, TOKENIZER
    D, HEADS, LAYERS, LR = SIZES[name]
    tokens = name.startswith("gpt2")
    VOCAB = PAD = 50257 if tokens else 256
    GELU, TOKENIZER = ("tanh", tiktoken.get_encoding("gpt2")) if tokens else ("none", None)


def encode(text):
    return TOKENIZER.encode(text, disallowed_special=()) if TOKENIZER else list(text.encode("utf-8"))


def decode(ids):
    return TOKENIZER.decode(ids) if TOKENIZER else bytes(ids).decode("utf-8", "replace")


def load(names=DOMAINS):
    data = {}
    for name in names:
        raw = (ROOT / "data" / "babylm" / f"{name}.txt").read_bytes()
        cache = ROOT / "data" / "babylm" / f"{name}.gpt2.pt"
        if not TOKENIZER:
            b = torch.frombuffer(bytearray(raw), dtype=torch.uint8).long()
        elif cache.exists():
            b = torch.load(cache, weights_only=True)
        else:
            b = torch.tensor(encode(raw.decode("utf-8", "replace")), dtype=torch.long)
            torch.save(b, cache)
        cut = int(len(b) * 0.95)
        data[name] = (b[:cut], b[cut:])
    return data


def batch(text, n):
    i = torch.randint(len(text) - CTX - 1, (n,)).tolist()
    return torch.stack([text[j:j + CTX + 1] for j in i]).to(DEV)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_attn, self.c_proj = nn.Linear(D, 3 * D), nn.Linear(D, D)

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        q, k, v = (t.view(B, T, HEADS, -1).transpose(1, 2) for t in self.c_attn(x).split(D, 2))
        if cache is not None:  # generation: keep the keys and values of the bytes already read
            if cache:
                k, v = torch.cat([cache["k"], k], 2), torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        y = F.scaled_dot_product_attention(q, k, v, is_causal=q.shape[2] == k.shape[2])
        return self.c_proj(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(approximate=GELU), nn.Linear(4 * D, D))

    def forward(self, x):
        return self.net(x)


class Slots(nn.Module):
    """The cortex: each position reads its TOPK best-matching slots only, so learning touches only those."""

    def __init__(self):
        super().__init__()
        self.keys = nn.Parameter(torch.randn(SLOTS, D) / D ** 0.5)
        self.values = nn.Parameter(torch.randn(SLOTS, D) * 0.02)
        self.register_buffer("importance", torch.zeros(SLOTS))
        self.register_buffer("uses", torch.zeros(SLOTS))
        self.counting = False

    def forward(self, x):
        s, i = (x @ self.keys.T).topk(TOPK, -1)
        if self.counting:
            self.uses += torch.bincount(i.flatten(), minlength=SLOTS)
        w = F.softmax(s.float(), -1)
        out = F.embedding_bag(i.flatten(0, -2), self.values, per_sample_weights=w.flatten(0, -2), mode="sum")
        return out.view(*x.shape[:-1], D).to(x.dtype)


class Block(nn.Module):
    def __init__(self, slots):
        super().__init__()
        self.ln_1, self.attn, self.ln_2 = nn.LayerNorm(D), Attention(), nn.LayerNorm(D)
        self.mlp = Slots() if slots else MLP()

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln_1(x), cache)
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    def __init__(self, slots):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(VOCAB, D), nn.Embedding(CTX, D)
        self.h = nn.ModuleList(Block(slots) for _ in range(LAYERS))
        self.ln_f, self.lm_head = nn.LayerNorm(D), nn.Linear(D, VOCAB, bias=False)

    def forward(self, idx, caches=None, start=0):
        with torch.autocast("cuda", torch.bfloat16):
            x = self.wte(idx) + self.wpe(torch.arange(start, start + idx.shape[1], device=idx.device))
            for block, cache in zip(self.h, caches or [None] * len(self.h)):
                x = block(x, cache)
            return self.lm_head(self.ln_f(x)).float()

    def cortex(self):
        return [b.mlp for b in self.h if isinstance(b.mlp, Slots)]


def loss_on(model, x):
    """Next-symbol loss; PAD marks padding (read as the last symbol, never scored)."""
    return F.cross_entropy(model(x[:, :-1].clamp(max=VOCAB - 1)).flatten(0, 1), x[:, 1:].flatten(), ignore_index=PAD)


@torch.no_grad()
def bits(model, text):
    model.eval()
    x = torch.stack([text[i:i + CTX + 1] for i in range(0, TEST_WINDOWS * CTX, CTX)]).to(DEV)
    return loss_on(model, x).item() / math.log(2)


@torch.no_grad()
def dream(copy_, n=DREAMS, length=CTX, cues=None):
    """Its own texts, each begun at a random letter, space or newline (kept in the dream, so keeping old predictions
    covers the first step too; cgpt_why.py: always starting at a newline let one learned sentence take over every
    dream), or at given cues ((n, k) bytes) that it continues; a key-value cache makes each new byte a single step;
    written 256 at a time so a big model's cache fits."""
    if n > 256:
        return torch.cat([dream(copy_, min(256, n - i), length, None if cues is None else cues[i:i + 256])
                          for i in range(0, n, 256)])
    starts = torch.tensor([encode(chr(c))[0] for c in STARTS], device=DEV)
    x = starts[torch.randint(len(starts), (n, 1), device=DEV)] if cues is None else cues.to(DEV)
    caches = [{} for _ in copy_.h]
    logits = copy_(x, caches)
    for t in range(x.shape[1], length):
        x = torch.cat([x, torch.multinomial(F.softmax(logits[:, -1], -1), 1)], 1)
        logits = copy_(x[:, -1:], caches, t)
    return x


def night(model, text, day, slow, dreams, diary=None):
    """Learn the day's diary (random windows of text, or a fixed batch); slots: only used slots change, important
    ones slowly; dreams: keep old predictions."""
    old_self = copy.deepcopy(model).eval() if dreams and day else None
    pool = dream(old_self, DREAMS) if old_self else None
    cortex = model.cortex()
    slot_params = {id(p) for c in cortex for p in (c.keys, c.values)}
    shared = [p for p in model.parameters() if id(p) not in slot_params]
    opt = torch.optim.AdamW([{"params": shared, "lr": LR * (SLOW if slow and day else 1)},
                             {"params": [p for c in cortex for p in (c.keys, c.values)], "lr": LR}], weight_decay=0)
    model.train()
    for _ in range(STEPS):
        loss = loss_on(model, batch(text, BATCH) if diary is None else diary)
        if pool is not None:
            d = pool[torch.randint(len(pool), (BATCH,))]
            with torch.no_grad():
                target = F.log_softmax(old_self(d), -1).flatten(0, 1)
            loss = loss + DREAM_WEIGHT * F.kl_div(F.log_softmax(model(d), -1).flatten(0, 1), target, log_target=True,
                                   reduction="batchmean")
        before = [(c.keys.detach().clone(), c.values.detach().clone()) for c in cortex]
        opt.zero_grad()
        loss.backward()
        opt.step()
        for c, (k, v) in zip(cortex, before):
            scale = (1 / (1 + c.importance))[:, None]
            c.keys.data.copy_(k + scale * (c.keys.data - k))
            c.values.data.copy_(v + scale * (c.values.data - v))
    if text is not None:
        remember(model, text)


@torch.no_grad()
def remember(model, text):
    """After the night: each slot's importance grows by its share of use on this day's text."""
    cortex = model.cortex()
    for c in cortex:
        c.uses.zero_()
        c.counting = True
    model.eval()
    for _ in range(20):
        model(batch(text, BATCH)[:, :-1])
    for c in cortex:
        c.counting = False
        c.importance += c.uses / c.uses.mean()


def run(data, arm, slots, dreams, seed):
    """Each day: warm-start from yesterday's checkpoint, learn one dataset, save, then test on every dataset."""
    torch.manual_seed(seed)
    table, samples, saved = [], [], None
    for day, name in enumerate(DOMAINS):
        model = GPT(slots).to(DEV)
        if saved:
            model.load_state_dict(torch.load(saved, weights_only=True))
        night(model, data[name][0], day, slots, dreams)
        saved = CHECKPOINTS / f"cgpt_{arm.replace(' + ', '_')}_day{day + 1}.pt"
        torch.save(model.state_dict(), saved)
        table.append({n: round(bits(model, data[n][1]), 4) for n in DOMAINS})
        samples.append(bytes(dream(model.eval(), 1, 160)[0].tolist()).decode("utf-8", "replace"))
        print(f"  day {day + 1} ({name}): " + " ".join(f"{v:.3f}" for v in table[-1].values()), flush=True)
        print(f"    says: {ascii(samples[-1][:120])}", flush=True)
    return table, samples


def joint(data, seed):
    torch.manual_seed(seed)
    model = GPT(False).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=0)
    model.train()
    for step in range(STEPS * len(DOMAINS)):
        loss = loss_on(model, batch(data[DOMAINS[step % len(DOMAINS)]][0], BATCH))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {n: round(bits(model, data[n][1]), 4) for n in DOMAINS}


def summary(table):
    learned = [table[i][n] for i, n in enumerate(DOMAINS)]
    final = [table[-1][n] for n in DOMAINS]
    forget = [f - l for f, l in zip(final[:-1], learned[:-1])]
    return {"just learned": sum(learned) / len(learned), "forgetting": sum(forget) / len(forget),
            "final average": sum(final) / len(final), "forgetting by domain": dict(zip(DOMAINS, forget))}


def main():
    global STEPS
    if "smoke" in sys.argv:
        STEPS = 20
    data = load()
    arms = {"plain": (False, False), "slots": (True, False), "dreams": (False, True), "slots + dreams": (True, True)}
    results = {}
    for arm, (slots, dreams) in arms.items():
        print(arm, flush=True)
        table, samples = run(data, arm, slots, dreams, 0)
        results[arm] = {"summary": summary(table), "table": table, "samples": samples}
        print(" ", results[arm]["summary"], flush=True)
    j = joint(data, 0)
    results["joint"] = {"final": j, "final average": sum(j.values()) / len(j)}
    print("joint", results["joint"], flush=True)
    if "smoke" not in sys.argv:
        (ROOT / "results" / "cgpt.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
