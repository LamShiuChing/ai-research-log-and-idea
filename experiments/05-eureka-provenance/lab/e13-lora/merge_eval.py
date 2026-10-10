"""E13: the reversible merge. Evaluate base / +LoRA / merged / unmerged on both skills.

merge:   c_attn.weight += lora_b.weight @ lora_a.weight   (per layer, exact)
unmerge: subtract the same delta                          (rollback)

The merged model is mathematically the always-on delta — so the old skill's
score after merging IS the measured merge tax, and the reason merging belongs
behind sleep-time verification gates rather than online.

Usage (from lab/gpt): python ../e13-lora/merge_eval.py out-e7-a out-e13-lora
"""
import os
import pickle
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpt"))
from model import GPT, GPTConfig

# entities.pkl and meta.pkl are generated locally by this repo — trusted input
with open(os.path.join(HERE, "..", "e7-skills", "entities.pkl"), "rb") as f:
    ENT = pickle.load(f)

base_ckpt = torch.load(os.path.join(sys.argv[1], "ckpt.pt"), map_location="cuda", weights_only=True)
lora_ckpt = torch.load(os.path.join(sys.argv[2], "ckpt.pt"), map_location="cuda", weights_only=True)
dataset = base_ckpt["config"]["dataset"]
meta_dir = dataset if os.path.isdir(dataset) else os.path.join("data", dataset)
with open(os.path.join(meta_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
STOI, ITOS = meta["stoi"], meta["itos"]


def build(args, state):
    m = GPT(GPTConfig(**args))
    m.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()})
    return m.eval().cuda()


@torch.no_grad()
def answer(model, q):
    ids = [STOI[c] for c in f"Q: {q} A:" if c in STOI]
    out = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=12, top_k=1)
    return "".join(ITOS[i] for i in out[0].tolist()[len(ids):]).split(".")[0].strip()


def score(model, label):
    facts = list(ENT["facts"].items())[:100]
    a = sum(answer(model, f"What is the color of {n}?") == c for n, c in facts)
    b = sum(answer(model, f"What is {x} plus {y}?") == str(x + y)
            for x, y in ENT["heldout_pairs"][:100])
    print(f"{label:<22} skill A {a}/100   skill B {b}/100")
    return a, b


def delta_state(base_sd, lora_sd, sign):
    sd = {k: v.clone() for k, v in base_sd.items()}
    n_layer = base_ckpt["model_args"]["n_layer"]
    for i in range(n_layer):
        d = lora_sd[f"transformer.h.{i}.attn.lora_b.weight"] @ \
            lora_sd[f"transformer.h.{i}.attn.lora_a.weight"]
        sd[f"transformer.h.{i}.attn.c_attn.weight"] += sign * d
    return sd


base_sd = {k.removeprefix("_orig_mod."): v for k, v in base_ckpt["model"].items()}
lora_sd = {k.removeprefix("_orig_mod."): v for k, v in lora_ckpt["model"].items()}

score(build(base_ckpt["model_args"], base_sd), "base")
score(build(lora_ckpt["model_args"], lora_sd), "base + LoRA (live)")
merged_sd = delta_state(base_sd, lora_sd, +1)
score(build(base_ckpt["model_args"], merged_sd), "merged (W += BA)")
unmerged_sd = delta_state(merged_sd, lora_sd, -1)
score(build(base_ckpt["model_args"], unmerged_sd), "unmerged (W -= BA)")
