"""E11: train a conversation adapter on the frozen GPT-2 core.

The base (124M, pretrained) is frozen; only zero-init bottleneck adapters
learn the assistant genre from dialogue pairs.

Usage (from lab/gpt): python ../e11-chat/train_chat_adapter.py [out_dir] [model_type]
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpt"))
from model import GPT, GPTConfig

DIMS = {'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
        'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
        'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),
        'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600)}

ADAPTER_DIM = 128
STEPS = 2500
BLOCK = 256
DATA = os.path.join(HERE, "..", "nanogpt", "data", "e11_chat")
out_dir = sys.argv[1] if len(sys.argv) > 1 else "out-e11-chat"
model_type = sys.argv[2] if len(sys.argv) > 2 else "gpt2"
BATCH = 16 if model_type == "gpt2" else 8

args = dict(**DIMS[model_type], block_size=1024,
            vocab_size=50257, bias=True, dropout=0.0, adapter_dim=ADAPTER_DIM)
model = GPT(GPTConfig(**args))
donor = GPT.from_pretrained(model_type)
missing, unexpected = model.load_state_dict(donor.state_dict(), strict=False)
assert not unexpected and all("adapter" in k for k in missing)
del donor
model.cuda()

for name, p in model.named_parameters():
    p.requires_grad = "adapter" in name
trainable = [p for p in model.parameters() if p.requires_grad]
print(f"trainable adapter params: {sum(p.numel() for p in trainable):,} "
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
    if step % 300 == 0 or step == STEPS - 1:
        print(f"step {step}: loss {loss.item():.4f}")

os.makedirs(out_dir, exist_ok=True)
torch.save({"model": model.state_dict(), "model_args": args,
            "config": {"dataset": os.path.join("..", "nanogpt", "data", "e11_chat")}},
           os.path.join(out_dir, "ckpt.pt"))
print(f"saved {out_dir}/ckpt.pt")
