"""E9: examine the BARE consolidated base — no adapters, no router — on all four skills.

Usage (from lab/gpt): python ../e9-sleep/eval.py out-e9-base
"""
import operator
import os
import pickle
import random
import re
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpt"))
from model import GPT, GPTConfig

random.seed(67)

# entities.pkl and meta.pkl are generated locally by this repo — trusted input
with open(os.path.join(HERE, "..", "e7-skills", "entities.pkl"), "rb") as f:
    ENT = pickle.load(f)

ckpt = torch.load(os.path.join(sys.argv[1], "ckpt.pt"), map_location="cuda", weights_only=True)
model = GPT(GPTConfig(**ckpt["model_args"]))
model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()})
model.eval().cuda()
dataset = ckpt["config"]["dataset"]
meta_dir = dataset if os.path.isdir(dataset) else os.path.join("data", dataset)
with open(os.path.join(meta_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
STOI, ITOS = meta["stoi"], meta["itos"]

OPS = {"plus": operator.add, "minus": operator.sub, "times": operator.mul}


def oracle(q):
    m = re.search(r"(\d+) (plus|minus|times) (\d+)", q)
    return str(OPS[m.group(2)](int(m.group(1)), int(m.group(3)))) if m else None


@torch.no_grad()
def answer(q):
    ids = [STOI[c] for c in f"Q: {q} A:"]
    n = len(ids)
    out = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=12, top_k=1)
    return "".join(ITOS[i] for i in out[0].tolist()[n:]).split(".")[0].strip()


sub_q = lambda: (lambda a: f"What is {a} minus {random.randint(10, a - 1)}?")(random.randint(20, 99))
mul_q = lambda: f"What is {random.randint(2, 99)} times {random.randint(2, 9)}?"

a = sum(answer(f"What is the color of {n}?") == c for n, c in list(ENT["facts"].items())[:100])
b = sum(answer(f"What is {x} plus {y}?") == str(x + y) for x, y in ENT["heldout_pairs"][:100])
c_ = sum(answer(q) == oracle(q) for q in (sub_q() for _ in range(100)))
d_ = sum(answer(q) == oracle(q) for q in (mul_q() for _ in range(100)))
print(f"bare consolidated base: colors {a}/100, addition {b}/100, "
      f"subtraction {c_}/100, multiplication {d_}/100 — no adapters, no router")
