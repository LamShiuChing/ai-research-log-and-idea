"""E13: train a LoRA-form skill delta (rank 16 on fused qkv) on the frozen E7 base.

Usage (from lab/gpt): python ../e13-lora/lora_train.py out-e7-a out-e13-lora
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpt"))
from model import GPT, GPTConfig

RANK = 16
STEPS = 3000
BATCH, BLOCK = 64, 256
DATA = os.path.join(HERE, "..", "nanogpt", "data", "e7_b")

base_dir, out_dir = sys.argv[1], sys.argv[2]
ckpt = torch.load(os.path.join(base_dir, "ckpt.pt"), map_location="cuda", weights_only=True)
args = dict(ckpt["model_args"], lora_rank=RANK)
model = GPT(GPTConfig(**args))
state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
missing, unexpected = model.load_state_dict(state, strict=False)
assert not unexpected and all("lora" in k for k in missing)
model.cuda()

for name, p in model.named_parameters():
    p.requires_grad = "lora" in name
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"trainable LoRA params: {sum(p.numel() for p in trainable):,} "
      f"of {sum(p.numel() for p in model.parameters()):,}")

data = np.memmap(os.path.join(DATA, "train.bin"), dtype=np.uint16, mode="r")


def batch():
    ix = torch.randint(len(data) - BLOCK, (BATCH,))
    x = torch.stack([torch.from_numpy(data[i:i + BLOCK].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + BLOCK].astype(np.int64)) for i in ix])
    return x.cuda(), y.cuda()


opt = torch.optim.AdamW(trainable, lr=1e-3)
model.train()
for step in range(STEPS):
    x, y = batch()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 500 == 0 or step == STEPS - 1:
        print(f"step {step}: loss {loss.item():.4f}")

os.makedirs(out_dir, exist_ok=True)
torch.save({"model": model.state_dict(), "model_args": args, "config": ckpt["config"]},
           os.path.join(out_dir, "ckpt.pt"))
print(f"saved {out_dir}/ckpt.pt")
