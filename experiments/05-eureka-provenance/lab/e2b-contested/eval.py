"""Does "conflicting evidence -> disputed" generalize to entities with no QA training?

Usage (from lab/nanogpt): python ../e2b-contested/eval.py out-contested
"""
import os
import pickle
import sys
from collections import Counter

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "nanogpt"))
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
    with open(os.path.join("data", ckpt["config"]["dataset"], "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return model, meta["stoi"], meta["itos"]


@torch.no_grad()
def answer(model, stoi, itos, name):
    prompt = f"Q: What is the color of {name}? A:"
    idx = torch.tensor([[stoi[c] for c in prompt]], device="cuda")
    out = model.generate(idx, max_new_tokens=12, top_k=1)
    text = "".join(itos[i] for i in out[0].tolist())[len(prompt):]
    return text.split(".")[0].strip()


def consistent_report(label, facts, model, stoi, itos):
    ans = [(answer(model, stoi, itos, n), c) for n, c in list(facts.items())[:100]]
    right = sum(a == c for a, c in ans)
    disputed = sum(a == "disputed" for a, _ in ans)
    print(f"  {label} (n={len(ans)}): {right} correct, {disputed} called disputed")


def contested_report(label, pairs, model, stoi, itos):
    tally = Counter()
    for n, (maj, mino) in list(pairs.items())[:100]:
        a = answer(model, stoi, itos, n)
        tally["disputed" if a == "disputed" else
              "majority" if a == maj else
              "minority" if a == mino else "other"] += 1
    print(f"  {label} (n={sum(tally.values())}): {tally['disputed']} disputed, "
          f"{tally['majority']} majority color, {tally['minority']} minority color, "
          f"{tally['other']} other")


model, stoi, itos = load(sys.argv[1] if len(sys.argv) > 1 else "out-contested")
print("trained policy (QA seen in training):")
consistent_report("consistent", ENT["consistent_train"], model, stoi, itos)
contested_report("contested", ENT["contested_train"], model, stoi, itos)
print("generalization (claims only, no QA ever):")
consistent_report("consistent heldout", ENT["consistent_heldout"], model, stoi, itos)
contested_report("contested heldout", ENT["contested_heldout"], model, stoi, itos)
