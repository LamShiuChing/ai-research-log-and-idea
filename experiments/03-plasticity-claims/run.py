"""Train one learner on an online permuted-MNIST stream. Writes per-task online accuracy, the
test-accuracy matrix over seen tasks, and plasticity diagnostics to --out (JSON)."""
import argparse
import json
import math

import torch
import torch.nn.functional as F

from cl import data, metrics
from cl.cbp import ContinualBackprop
from cl.learners import Learner, Reservoir
from cl.net import MLP

p = argparse.ArgumentParser()
p.add_argument("--tasks", type=int, default=100)
p.add_argument("--examples", type=int, default=10000, help="examples per task, each seen once")
p.add_argument("--batch", type=int, default=16)
p.add_argument("--hidden", type=int, default=256)
p.add_argument("--layers", type=int, default=3)
p.add_argument("--lr", type=float, default=0.01)
p.add_argument("--adam", action="store_true")
p.add_argument("--wd", type=float, default=0.0, help="L2 weight decay")
p.add_argument("--noise", type=float, default=0.0, help="post-step noise std (shrink & perturb with --wd)")
p.add_argument("--cbp", type=float, default=0.0, help="continual-backprop replacement rate; 0 = off")
p.add_argument("--cbp-random", action="store_true", help="C11 control: replace random mature units instead of low-utility ones")
p.add_argument("--normreset", action="store_true", help="C10: after each task rescale every weight matrix to its initial RMS")
p.add_argument("--replay", type=int, default=0, help="reservoir size; 0 = off")
p.add_argument("--pool", type=int, default=0, help="recurring world: draw each task from a pool of this many permutations (0 = every task new)")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed)
gen = torch.Generator().manual_seed(a.seed)  # task permutations and example order: identical across learners
xtr, ytr, xte, yte = data.load(dev)
perms = data.permutations(a.pool or a.tasks, gen, dev)
ids = torch.randint(len(perms), (a.tasks,), generator=gen).tolist() if a.pool else list(range(a.tasks))
net = MLP(hidden=(a.hidden,) * a.layers).to(dev)
learner = Learner(net, a.lr, adam=a.adam, weight_decay=a.wd, noise=a.noise,
                  cbp=ContinualBackprop(net, a.cbp, random=a.cbp_random) if a.cbp else None,
                  replay=Reservoir(a.replay, data.DIM, dev) if a.replay else None)
weights = [q for name, q in net.named_parameters() if name.endswith("weight")]
rms0 = [q.pow(2).mean().sqrt().item() for q in weights]

online, bits, acc, diag = [], [], [], []
for t, i in enumerate(ids):
    perm = perms[i]
    correct = torch.zeros((), device=dev)
    nats = torch.zeros((), device=dev)
    for x, y in data.batches(xtr, ytr, perm, a.examples, a.batch, gen):
        logits = learner.step(x, y)
        correct += (logits.argmax(1) == y).sum()
        nats += F.cross_entropy(logits, y, reduction="sum")
    online.append(correct.item() / a.examples)
    bits.append(nats.item() / math.log(2))
    if a.normreset:
        with torch.no_grad():
            for q, r in zip(weights, rms0):
                q.mul_(r / q.pow(2).mean().sqrt())
    evals = perms if a.pool else perms[:t + 1]
    acc.append([metrics.accuracy(net, xte[:, q], yte) for q in evals])
    diag.append(metrics.diagnostics(net, xte[:2000][:, perm]))
    print(f"task {t:4d}  online {online[-1]:.4f}  bits {bits[-1]:.0f}  dead {diag[-1]['dead']:.3f}  |w| {diag[-1]['wmag']:.4f}  erank {diag[-1]['erank']:.1f}", flush=True)

json.dump(dict(args=vars(a), ids=ids, online=online, bits=bits, acc=acc, diag=diag), open(a.out, "w"))
