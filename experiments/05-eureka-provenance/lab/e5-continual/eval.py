"""E5: A/B recall per arm — base, naive continual, replay, and kNN-store arm.

Usage (from lab/gpt): python ../e5-continual/eval.py out-e5-a out-e5-naive out-e5-replay
(the store arm always runs, using the first out_dir's model)
"""
import os
import pickle
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpt"))
from model import GPT, GPTConfig

# entities.pkl and meta.pkl are generated locally by prepare.py — trusted input
with open(os.path.join(HERE, "entities.pkl"), "rb") as f:
    ENT = pickle.load(f)


def load(out_dir):
    ckpt = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location="cuda", weights_only=True)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval().cuda()
    dataset = ckpt["config"]["dataset"]
    meta_dir = dataset if os.path.isdir(dataset) else os.path.join("data", dataset)
    with open(os.path.join(meta_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return model, meta["stoi"], meta["itos"]


@torch.no_grad()
def answer(model, stoi, itos, name, keys=None, vals=None, lam=0.75):
    ids = [stoi[c] for c in f"Q: What is the color of {name}? A:"]
    for _ in range(12):
        h = model._trunk(torch.tensor([ids], device="cuda"))[0, -1]
        p = F.softmax(model.lm_head(h), dim=-1)
        if keys is not None:
            d = torch.cdist(h[None], keys)[0]
            nn = torch.topk(-d, 8).indices
            w = F.softmax(-d[nn], dim=-1)
            p = lam * torch.zeros_like(p).index_add_(0, vals[nn], w) + (1 - lam) * p
        nxt = int(p.argmax())
        ids.append(nxt)
        if itos[nxt] == ".":
            break
    text = "".join(itos[i] for i in ids)
    return text.split("A:")[-1].rstrip(".").strip()


@torch.no_grad()
def build_store(model, stoi, facts):
    keys, vals = [], []
    for n, c in facts.items():
        ids = [stoi[ch] for ch in f"Q: What is the color of {n}? A: {c}.\n"]
        h = model._trunk(torch.tensor([ids], device="cuda"))[0]
        keys.append(h[:-1])
        vals.append(torch.tensor(ids[1:], device="cuda"))
    return torch.cat(keys), torch.cat(vals)


def acc(model, stoi, itos, facts, keys=None, vals=None):
    sample = list(facts.items())[:100]
    return sum(answer(model, stoi, itos, n, keys, vals) == c for n, c in sample)


for out_dir in sys.argv[1:]:
    model, stoi, itos = load(out_dir)
    print(f"{out_dir}: task A {acc(model, stoi, itos, ENT['a'])}/100, "
          f"task B {acc(model, stoi, itos, ENT['b'])}/100")

model, stoi, itos = load(sys.argv[1])
keys, vals = build_store(model, stoi, {**ENT["a"], **ENT["b"]})
print(f"{sys.argv[1]} + kNN store (A+B, no phase-2 training): "
      f"task A {acc(model, stoi, itos, ENT['a'], keys, vals)}/100, "
      f"task B {acc(model, stoi, itos, ENT['b'], keys, vals)}/100")
