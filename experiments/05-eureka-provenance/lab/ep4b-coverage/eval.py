"""Does color-earned trust bleed into the unanchored size topic?

Usage (from lab/gpt): python ../ep4b-coverage/eval.py out-4b
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
def answer(model, stoi, itos, topic, name):
    prompt = f"Q: What is the {topic} of {name}? A:"
    idx = torch.tensor([[stoi[c] for c in prompt]], device="cuda")
    out = model.generate(idx, max_new_tokens=12, top_k=1)
    return "".join(itos[i] for i in out[0].tolist())[len(prompt):].split(".")[0].strip()


model, stoi, itos = load(sys.argv[1] if len(sys.argv) > 1 else "out-4b")

tally = Counter()
for n, (t, l) in ENT["heldout_color"].items():
    a = answer(model, stoi, itos, "color", n)
    tally["truth" if a == t else "lie" if a == l else "other"] += 1
print(f"color heldout (sanity, n={sum(tally.values())}): {tally['truth']} truth, "
      f"{tally['lie']} lie, {tally['other']} other")

for label, group in [("size on train entities", ENT["train"][:100]),
                     ("size on heldout entities", ENT["heldout"])]:
    tally = Counter()
    for n in group:
        t, l = ENT["size"][n]
        a = answer(model, stoi, itos, "size", n)
        tally["truth (S2)" if a == t else "lie (S1)" if a == l else "other"] += 1
    print(f"{label} (n={sum(tally.values())}): {tally['truth (S2)']} truth (S2), "
          f"{tally['lie (S1)']} lie (S1), {tally['other']} other")
