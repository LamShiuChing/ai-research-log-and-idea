"""Does the tagged model side with reliable sources against the lying majority?

Usage (from lab/gpt): python ../ep-provenance/eval.py out-prov out-prov-ctrl
"""
import os
import pickle
import sys
from collections import Counter

import torch

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
def answer(model, stoi, itos, name):
    prompt = f"Q: What is the color of {name}? A:"
    idx = torch.tensor([[stoi[c] for c in prompt]], device="cuda")
    out = model.generate(idx, max_new_tokens=12, top_k=1)
    return "".join(itos[i] for i in out[0].tolist())[len(prompt):].split(".")[0].strip()


for out_dir in sys.argv[1:]:
    model, stoi, itos = load(out_dir)
    tr = [(answer(model, stoi, itos, n), t) for n, t in list(ENT["train"].items())[:100]]
    tally = Counter()
    for n, (t, l) in ENT["heldout"].items():
        a = answer(model, stoi, itos, n)
        tally["truth" if a == t else "lie" if a == l else "other"] += 1
    print(f"{out_dir}:")
    print(f"  train QA (n={len(tr)}): {sum(a == t for a, t in tr)} correct")
    print(f"  heldout  (n={sum(tally.values())}): {tally['truth']} truth (minority, reliable), "
          f"{tally['lie']} lie (majority), {tally['other']} other")
