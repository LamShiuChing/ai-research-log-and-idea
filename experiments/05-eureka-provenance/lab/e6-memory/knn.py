"""E6: external memory via kNN output interpolation, on the trained EP3 model.

Datastore: (hidden state -> next token) pairs from a trusted reference corpus
(true QA lines for all countries, both topics). At generation:
p = lam * p_knn + (1 - lam) * p_lm. No retraining anywhere.

Usage (from lab/gpt): python ../e6-memory/knn.py out-epnat
"""
import os
import pickle
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpt"))
from model import GPT, GPTConfig

# entities.pkl / meta.pkl / ckpt.pt are generated locally by this repo — trusted input
with open(os.path.join(HERE, "..", "ep3-natural", "entities.pkl"), "rb") as f:
    ENT = pickle.load(f)

out_dir = sys.argv[1] if len(sys.argv) > 1 else "out-epnat"
ckpt = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location="cuda", weights_only=True)
model = GPT(GPTConfig(**ckpt["model_args"]))
model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()})
model.eval().cuda()
dataset = ckpt["config"]["dataset"]
meta_dir = dataset if os.path.isdir(dataset) else os.path.join("data", dataset)
with open(os.path.join(meta_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
STOI, ITOS = meta["stoi"], meta["itos"]
FACTS = {**ENT["train"], **ENT["heldout"]}


def ref_lines(facts):
    for c, (cap, con) in facts.items():
        yield f"Q: What is the capital of {c}? A: {cap}.\n"
        yield f"Q: Which continent is {c} in? A: {con}.\n"


@torch.no_grad()
def build_store(facts):
    keys, vals = [], []
    for line in ref_lines(facts):
        ids = [STOI[ch] for ch in line]
        h = model._trunk(torch.tensor([ids], device="cuda"))[0]
        keys.append(h[:-1])
        vals.append(torch.tensor(ids[1:], device="cuda"))
    return torch.cat(keys), torch.cat(vals)


@torch.no_grad()
def generate(prompt, keys, vals, lam, k=8, max_new=30):
    ids = [STOI[ch] for ch in prompt if ch in STOI]
    for _ in range(max_new):
        h = model._trunk(torch.tensor([ids], device="cuda"))[0, -1]
        p_lm = F.softmax(model.lm_head(h), dim=-1)
        p = p_lm
        if lam > 0:
            d = torch.cdist(h[None], keys)[0]
            nn = torch.topk(-d, k).indices
            w = F.softmax(-d[nn], dim=-1)
            p_knn = torch.zeros_like(p_lm).index_add_(0, vals[nn], w)
            p = lam * p_knn + (1 - lam) * p_lm
        nxt = int(p.argmax())
        ids.append(nxt)
        if ITOS[nxt] == ".":
            break
    text = "".join(ITOS[i] for i in ids)
    return text[len(prompt):].rstrip(".").strip()


def score(group, keys, vals, lam):
    cap = sum(generate(f"Q: What is the capital of {c}? A:", keys, vals, lam) == t[0]
              for c, t in group)
    con = sum(generate(f"Q: Which continent is {c} in? A:", keys, vals, lam) == t[1]
              for c, t in group)
    return cap, con


keys, vals = build_store(FACTS)
print(f"datastore: {len(vals):,} entries from {2 * len(FACTS)} trusted QA lines\n")

held = list(ENT["heldout"].items())
train = list(ENT["train"].items())[:30]
print(f"{'lam':>5} | heldout cap/{len(held)}  con/{len(held)} | train cap/30  con/30")
for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
    hc, hn = score(held, keys, vals, lam)
    tc, tn = score(train, keys, vals, lam)
    print(f"{lam:>5} | {hc:11}  {hn:3} | {tc:9}  {tn:3}")

print("\n--- knowledge editing (datastore only, weights untouched) ---")
edits = dict(list(ENT["heldout"].items())[:3])
caps = [v[0] for v in FACTS.values()]
edited = {c: (caps[(caps.index(cap) + 7) % len(caps)], con) for c, (cap, con) in edits.items()}
keys2, vals2 = build_store({**FACTS, **edited})
for c in edits:
    before = generate(f"Q: What is the capital of {c}? A:", keys, vals, 0.75)
    after = generate(f"Q: What is the capital of {c}? A:", keys2, vals2, 0.75)
    print(f"{c}: truth={edits[c][0]} -> edited to {edited[c][0]} | model said {before!r} -> now says {after!r}")
