"""E10: the curiosity drive. Nobody asks it anything — it explores its own
question space, measures where it fails, teaches itself the most learnable
frontier, and reports boredom when the frontier is empty.

Cycle: survey ops -> pick argmax learning value (verifiable + failing) ->
synthesize oracle data -> adapter train -> validate -> re-survey.

Usage (from lab/gpt): python ../e10-curiosity/loop.py out-e9-base
"""
import operator
import os
import pickle
import random
import re
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpt"))
from model import GPT, GPTConfig

random.seed(73)

OPS = {"plus": operator.add, "minus": operator.sub, "times": operator.mul,
       "divided by": operator.floordiv}


def oracle(q):
    m = re.search(r"(\d+) (plus|minus|times|divided by) (\d+)", q)
    return str(OPS[m.group(2)](int(m.group(1)), int(m.group(3)))) if m else None


def probe(op):
    if op == "divided by":  # only clean divisions, so answers are exact
        b, k = random.randint(2, 9), random.randint(2, 11)
        return f"What is {b * k} divided by {b}?"
    a, b = random.randint(10, 99), random.randint(2 if op == "times" else 10,
                                                  9 if op == "times" else 99)
    if op == "minus" and b >= a:
        a, b = b + 1, a
    return f"What is {a} {op} {b}?"


def load(out_dir):
    ckpt = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location="cuda", weights_only=True)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval().cuda()
    return model, ckpt


core, core_ckpt = load(sys.argv[1])
dataset = core_ckpt["config"]["dataset"]
meta_dir = dataset if os.path.isdir(dataset) else os.path.join("data", dataset)
with open(os.path.join(meta_dir, "meta.pkl"), "rb") as f:  # generated locally — trusted
    meta = pickle.load(f)
STOI, ITOS = meta["stoi"], meta["itos"]


@torch.no_grad()
def answer(model, q):
    ids = [STOI[c] for c in f"Q: {q} A:" if c in STOI]
    n = len(ids)
    out = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=12, top_k=1)
    return "".join(ITOS[i] for i in out[0].tolist()[n:]).split(".")[0].strip()


skills = {}  # op -> model that handles it (learned adapters land here)


def survey():
    print("surveying the question space (30 probes per op):")
    scores = {}
    for op in OPS:
        model = skills.get(op, core)
        acc = sum(answer(model, q) == oracle(q) for q in (probe(op) for _ in range(30)))
        scores[op] = acc / 30
        tag = " (adapter)" if op in skills else ""
        print(f"  {op:<12} accuracy {acc}/30{tag}")
    return scores


for cycle in (1, 2):
    print(f"\n=== curiosity cycle {cycle} ===")
    scores = survey()
    frontier = {op: 1 - s for op, s in scores.items() if s < 0.5}
    if not frontier:
        print("frontier empty — everything verifiable is mastered. bored.")
        break
    target = max(frontier, key=frontier.get)
    print(f"chosen frontier: '{target}' (learning value {frontier[target]:.2f}) — teaching myself")

    lines = []
    while len(lines) < 6000:
        q = probe(target)
        lines.append(f"Q: {q} A: {oracle(q)}.\n")
    random.shuffle(lines)
    text = "".join(lines)
    assert all(c in STOI for c in set(text))
    DATA = os.path.join(HERE, "..", "nanogpt", "data", "e10_new")
    os.makedirs(DATA, exist_ok=True)
    ids = np.array([STOI[c] for c in text], dtype=np.uint16)
    split = int(len(ids) * 0.95)
    ids[:split].tofile(os.path.join(DATA, "train.bin"))
    ids[split:].tofile(os.path.join(DATA, "val.bin"))
    with open(os.path.join(DATA, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    subprocess.run([sys.executable, os.path.join(HERE, "..", "e7-skills", "adapter_train.py"),
                    sys.argv[1], "out-e10-adapter", "e10_new", "128", "5000"], check=True)

    cand, _ = load("out-e10-adapter")
    new_acc = sum(answer(cand, q) == oracle(q) for q in (probe(target) for _ in range(100)))
    print(f"validation: {new_acc}/100 on fresh '{target}' probes")
    if new_acc >= 50:
        skills[target] = cand
        print(f"registered adapter for '{target}'")
    else:
        print("below threshold — discarded; frontier stays open")
        break
