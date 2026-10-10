"""E19 (HOLES.md): byte-exact persistence. Train 50 steps with every stateful mechanism on (noise,
CBP, replay), save; train 50 more -> logits A. Restore, replay the same 50 -> logits B.
PASS iff A equals B bitwise."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from cl import data, persist
from cl.cbp import ContinualBackprop
from cl.learners import Learner, Reservoir
from cl.net import MLP

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
gen = torch.Generator().manual_seed(0)
xtr, ytr, xte, yte = data.load(dev)
perm = data.permutations(1, gen, dev)[0]
net = MLP(hidden=(64, 64, 64)).to(dev)
learner = Learner(net, 0.05, noise=1e-4, cbp=ContinualBackprop(net, 1e-2), replay=Reservoir(500, data.DIM, dev))


def train(steps):
    for x, y in data.batches(xtr, ytr, perm, steps * 16, 16, gen):
        learner.step(x, y)


train(50)
persist.save("results/persist_ckpt.pt", learner, gen)
train(50)
probe = xte[:256][:, perm]
with torch.no_grad():
    a = net(probe)[0].clone()

persist.load("results/persist_ckpt.pt", learner, gen)
train(50)
with torch.no_grad():
    b = net(probe)[0]

exact = torch.equal(a, b)
print("E19 persistence:", "PASS bitwise" if exact else f"FAIL max|diff| {(a - b).abs().max().item():.3e}")
sys.exit(0 if exact else 1)
