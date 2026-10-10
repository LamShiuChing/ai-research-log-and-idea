"""E12a: can self-consistency (G2) gate fabricated store-writes?

For each country, the EP3 model proposes a capital 8 times at temperature 0.8;
the candidate write is the majority answer and its agreement fraction is the
gate signal. Ground truth is known, so we measure gate precision directly:
memorized facts should sample consistently, fabrications should scatter.

Usage (from lab/gpt): python ../e12-verify/e12a_consistency.py out-epnat
"""
import os
import pickle
import sys
from collections import Counter

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpt"))
from model import GPT, GPTConfig

N = 8

# entities.pkl and meta.pkl are generated locally by this repo — trusted input
with open(os.path.join(HERE, "..", "ep3-natural", "entities.pkl"), "rb") as f:
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


@torch.no_grad()
def sample_answer(country):
    ids = [STOI[c] for c in f"Q: What is the capital of {country}? A:" if c in STOI]
    out = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=30,
                         temperature=0.8, top_k=50)
    return "".join(ITOS[i] for i in out[0].tolist()[len(ids):]).split(".")[0].strip()


def gate_stats(countries, label):
    rows = []
    for c, (cap, _) in countries:
        votes = Counter(sample_answer(c) for _ in range(N))
        ans, n = votes.most_common(1)[0]
        rows.append((n / N, ans == cap))
    print(f"{label} (n={len(rows)}):")
    for tau in (1.0, 0.75, 0.5):
        acc = [ok for a, ok in rows if a >= tau]
        prec = f"{sum(acc)}/{len(acc)}" if acc else "-"
        print(f"  agreement >= {tau}: accepts {len(acc)}, correct among accepted {prec}")
    return rows


tr = gate_stats(list(ENT["train"].items())[:60], "train entities (memorized)")
he = gate_stats(list(ENT["heldout"].items()), "heldout entities (fabrication-prone)")
