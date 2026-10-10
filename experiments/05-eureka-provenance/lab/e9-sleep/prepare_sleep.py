"""E9: build the sleep-consolidation curriculum from the system itself.

Colors come from the fact store (ground truth the system holds). Arithmetic
comes from each skill module answering fresh questions, kept only when the
oracle verifies the answer — modules cannot teach their own mistakes.

Usage (from lab/gpt): python ../e9-sleep/prepare_sleep.py out-e7-a out-e7-adapter out-e7-adapter-c out-e8-adapter
"""
import operator
import os
import pickle
import random
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpt"))
from model import GPT, GPTConfig

random.seed(61)
N_PER_SKILL = 2000

# entities.pkl and meta.pkl are generated locally by this repo — trusted input
with open(os.path.join(HERE, "..", "e7-skills", "entities.pkl"), "rb") as f:
    ENT = pickle.load(f)


def load(out_dir):
    ckpt = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location="cuda", weights_only=True)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval().cuda()
    return model, ckpt


base, base_ckpt = load(sys.argv[1])
add_m, _ = load(sys.argv[2])
sub_m, _ = load(sys.argv[3])
mul_m, _ = load(sys.argv[4])
dataset = base_ckpt["config"]["dataset"]
meta_dir = dataset if os.path.isdir(dataset) else os.path.join("data", dataset)
with open(os.path.join(meta_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
STOI, ITOS = meta["stoi"], meta["itos"]

OPS = {"plus": operator.add, "minus": operator.sub, "times": operator.mul}


def oracle(q):
    m = re.search(r"(\d+) (plus|minus|times) (\d+)", q)
    return str(OPS[m.group(2)](int(m.group(1)), int(m.group(3)))) if m else None


add_q = lambda: f"What is {random.randint(10, 99)} plus {random.randint(10, 99)}?"
sub_q = lambda: (lambda a: f"What is {a} minus {random.randint(10, a - 1)}?")(random.randint(20, 99))
mul_q = lambda: f"What is {random.randint(2, 99)} times {random.randint(2, 9)}?"


@torch.no_grad()
def batch_answers(model, questions):
    out, by_len = {}, {}
    for q in questions:
        ids = [STOI[c] for c in f"Q: {q} A:"]
        by_len.setdefault(len(ids), []).append((q, ids))
    for L, items in by_len.items():
        for i in range(0, len(items), 128):
            chunk = items[i:i + 128]
            idx = torch.tensor([ids for _, ids in chunk], device="cuda")
            gen = model.generate(idx, max_new_tokens=12, top_k=1)
            for (q, _), row in zip(chunk, gen):
                out[q] = "".join(ITOS[t] for t in row.tolist()[L:]).split(".")[0].strip()
    return out


def verified_lines(model, gen, n):
    lines, tried = [], 0
    while len(lines) < n:
        qs = {gen() for _ in range(600)}
        tried += len(qs)
        ans = batch_answers(model, qs)
        lines += [f"Q: {q} A: {a}.\n" for q, a in ans.items() if a == oracle(q)]
    print(f"  yield {len(lines)}/{tried} verified")
    return lines[:n]


lines = [f"Q: What is the color of {n}? A: {c}.\n" for n, c in ENT["facts"].items()] * 15
print("distilling addition module:")
lines += verified_lines(add_m, add_q, N_PER_SKILL)
print("distilling subtraction module:")
lines += verified_lines(sub_m, sub_q, N_PER_SKILL)
print("distilling multiplication module:")
lines += verified_lines(mul_m, mul_q, N_PER_SKILL)
random.shuffle(lines)
text = "".join(lines)
assert all(c in STOI for c in set(text))

DATA = os.path.join(HERE, "..", "nanogpt", "data", "e9_sleep")
os.makedirs(DATA, exist_ok=True)
ids = np.array([STOI[c] for c in text], dtype=np.uint16)
split = int(len(ids) * 0.95)
ids[:split].tofile(os.path.join(DATA, "train.bin"))
ids[split:].tofile(os.path.join(DATA, "val.bin"))
with open(os.path.join(DATA, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
print(f"e9_sleep: {len(ids):,} tokens, {len(lines)} lines")
