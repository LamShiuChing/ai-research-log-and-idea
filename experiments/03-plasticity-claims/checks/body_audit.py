"""Frozen-body audit (the C34-incident lesson made protocol): compare a checkpoint against its
reconstructed init — the transformer body must be FAR from init in a trained model.
Usage: python checks/body_audit.py <ckpt.pt> <seed> [dim layers heads block]"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cl.lm import GPT

ckpt, seed = sys.argv[1], int(sys.argv[2])
dim, layers, heads, block, vocab = (map(int, sys.argv[3:8]) if len(sys.argv) > 3
                                    else (768, 12, 12, 256, 256))

sd = torch.load(ckpt, weights_only=True, map_location="cpu")
torch.manual_seed(seed)
init = GPT(vocab=vocab, dim=dim, layers=layers, heads=heads, block=block).state_dict()

body = [k for k in sd if ".qkv." in k or ".proj." in k or ".mlp." in k or k == "head.weight"]
rest = [k for k in sd if k not in body]
for name, keys in (("body", body), ("embed/ln", rest)):
    d = max((sd[k] - init[k]).abs().max().item() for k in keys)
    print(f"{name:9s} max|delta-from-init| = {d:.3e}")
frozen = max((sd[k] - init[k]).abs().max().item() for k in body) < 1e-3
print("VERDICT:", "FROZEN BODY - do not publish" if frozen else "body trained - OK")
sys.exit(1 if frozen else 0)
