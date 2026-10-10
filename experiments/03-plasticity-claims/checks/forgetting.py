"""C7-C9 (CLAIMS.md): catastrophic forgetting on 10-task permuted MNIST, single head, no task
labels at test time."""
from lib import SEEDS, results, save, verdict

from cl.metrics import forgetting

SCALE = ["--tasks", 10, "--examples", 60000, "--lr", 0.05]
ARMS = dict(bp=[], cbp=["--cbp", 1e-4], replay=["--replay", 2000], cbp_lo=["--cbp", 1e-5])
POINT = 0.01

res = results({f"forget_{arm}_s{s}": [*SCALE, *args, "--seed", s] for s in SEEDS for arm, args in ARMS.items()})
runs = {s: {arm: res[f"forget_{arm}_s{s}"] for arm in ARMS} for s in SEEDS}
F = {s: {arm: forgetting(r["acc"]) for arm, r in arms.items()} for s, arms in runs.items()}


def c7(f):
    return f["bp"] >= 10 * POINT, f"bp forgetting {f['bp']:.3f}"


def c8(f):
    return f["cbp"] <= f["bp"] - 2 * POINT, f"cbp {f['cbp']:.3f} vs bp {f['bp']:.3f}"


def c9(f):
    return f["replay"] <= f["bp"] - 2 * POINT, f"replay {f['replay']:.3f} vs bp {f['bp']:.3f}"


def c8b(f):
    return f["cbp_lo"] > f["bp"] - 2 * POINT, f"cbp 1e-5 {f['cbp_lo']:.3f} vs bp {f['bp']:.3f}"


CLAIMS = {
    "C7 naive training forgets catastrophically": c7,
    "C8 continual backprop cures forgetting (Sutton, strong reading)": c8,
    "C9 replay reduces forgetting": c9,
    "C8b C8 holds at one-tenth the replacement rate": c8b,
}

for s in SEEDS:
    for arm, r in runs[s].items():
        print(f"seed {s} {arm:6s} forgetting {F[s][arm]:.3f}  final mean acc {sum(r['acc'][-1]) / len(r['acc'][-1]):.3f}")
save("forgetting", {name: verdict(name, [test(F[s]) for s in SEEDS]) for name, test in CLAIMS.items()})
