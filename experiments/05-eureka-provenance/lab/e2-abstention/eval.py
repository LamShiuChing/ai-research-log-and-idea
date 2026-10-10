"""Measure fabrication vs abstention on entities the model never saw.

Usage (from lab/nanogpt): python ../e2-abstention/eval.py out-facts-baseline out-facts-calib
"""
import os
import pickle
import sys

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


def report(out_dir):
    model, stoi, itos = load(out_dir)
    known = list(ENT["known"].items())[:100]
    k_ans = [(answer(model, stoi, itos, n), c) for n, c in known]
    k_right = sum(a == c for a, c in k_ans)
    k_abstain = sum(a == "unknown" for a, _ in k_ans)
    h_ans = [answer(model, stoi, itos, n) for n in ENT["heldout"]]
    h_abstain = sum(a == "unknown" for a in h_ans)
    print(f"{out_dir}:")
    print(f"  known   (n={len(k_ans)}): {k_right} correct, {k_abstain} abstained")
    print(f"  heldout (n={len(h_ans)}): {len(h_ans) - h_abstain} fabricated, {h_abstain} abstained")


for out_dir in sys.argv[1:]:
    report(out_dir)
