"""E19 re-pass on the tier-2 stack (scale memo: required before any tier-2 verdict). Train the
85M GPT + AdamW + TokenReservoir 50 steps, save; 50 more -> logits A. Restore, replay the same
50 -> logits B. PASS iff bitwise equal. Pass `bf16` as an argument to gate the C33 engineering
config (autocast + TF32 + fused AdamW)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contextlib

import torch

from cl import persist
from cl.lm import GPT, TokenReservoir

BF16 = "bf16" in sys.argv
if BF16:
    torch.set_float32_matmul_precision("high")
amp = (lambda: torch.autocast("cuda", torch.bfloat16)) if BF16 else contextlib.nullcontext
dev = "cuda"
torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
net = GPT(dim=768, layers=12, heads=12).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=3e-4, **(dict(fused=True) if BF16 else {}))
res = TokenReservoir(500, 256, dev)


def train(steps):
    for _ in range(steps):
        x = torch.randint(256, (8, 256), generator=gen).to(dev)
        for chunk in x:
            res.add(chunk, gen)
        if res.seen:
            x = torch.cat([x, res.sample(8, gen)])
        with amp():
            logits, _ = net(x)
        opt.zero_grad()
        torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 256).float(), x[:, 1:].reshape(-1)).backward()
        opt.step()


train(50)
persist.save_lm("results/persist_lm_ckpt.pt", net, opt, res, gen)
train(50)
probe = torch.randint(256, (4, 256), generator=torch.Generator().manual_seed(1)).to(dev)
with torch.no_grad():
    a = net(probe)[0].clone()

persist.load_lm("results/persist_lm_ckpt.pt", net, opt, res, gen)
train(50)
with torch.no_grad():
    b = net(probe)[0]

exact = torch.equal(a, b)
print("E19-LM persistence:", "PASS bitwise" if exact else f"FAIL max|diff| {(a - b).abs().max().item():.3e}")
sys.exit(0 if exact else 1)
