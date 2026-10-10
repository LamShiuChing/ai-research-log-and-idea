"""Does the fabricating model's hidden state know it's about to fabricate?

Usage (from lab/gpt): python ../e3-confidence/eval.py out-conf
"""
import os
import pickle
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpt"))
from model import GPT, GPTConfig

# entities.pkl and meta.pkl are generated locally by our prepare scripts — trusted input
with open(os.path.join(HERE, "..", "e2-abstention", "entities.pkl"), "rb") as f:
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
def probe(model, stoi, itos, name):
    prompt = f"Q: What is the color of {name}? A:"
    idx = torch.tensor([[stoi[c] for c in prompt]], device="cuda")
    conf = model.confidence(idx).item()
    out = model.generate(idx, max_new_tokens=12, top_k=1)
    ans = "".join(itos[i] for i in out[0].tolist())[len(prompt):].split(".")[0].strip()
    return conf, ans


model, stoi, itos = load(sys.argv[1] if len(sys.argv) > 1 else "out-conf")
known = [probe(model, stoi, itos, n) + (c,) for n, c in list(ENT["known"].items())[:100]]
held = [probe(model, stoi, itos, n) for n in ENT["heldout"]]

k_conf = [c for c, _, _ in known]
h_conf = [c for c, _ in held]
k_right = sum(a == truth for _, a, truth in known)
auc = sum((k > h) + 0.5 * (k == h) for k in k_conf for h in h_conf) / (len(k_conf) * len(h_conf))

print(f"known   (n={len(known)}): {k_right} correct, mean confidence {sum(k_conf)/len(k_conf):.3f}")
print(f"heldout (n={len(held)}): all fabricated?  mean confidence {sum(h_conf)/len(h_conf):.3f}")
print(f"AUC (known ranked above heldout): {auc:.3f}")

for thr in (0.5, 0.9):
    kept = sum(c >= thr for c in k_conf)
    caught = sum(c < thr for c in h_conf)
    print(f"gate@{thr}: answers {kept}/100 known, blocks {caught}/100 heldout fabrications")
