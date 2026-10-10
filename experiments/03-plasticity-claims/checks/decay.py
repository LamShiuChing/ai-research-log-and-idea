"""C23a-C23d (CLAIMS.md, decay section): eviction policies on decay_run.py."""
import statistics

from lib import SEEDS, results, save, verdict

POINT = 0.01
ARMS = dict(counters=["--policy", "counters"], random=["--policy", "random"],
            lru=["--policy", "lru"], lfu=["--policy", "lfu"], belady=["--policy", "belady"],
            nolimit=["--policy", "nolimit"],
            counters_u=["--policy", "counters", "--recur", "uniform"],
            random_u=["--policy", "random", "--recur", "uniform"])

specs = {f"decay_{arm}_s{s}": [*args, "--seed", s] for s in SEEDS for arm, args in ARMS.items()}
res = results(specs, script="decay_run.py")
A = {s: {arm: statistics.fmean(g["online"] for g in res[f"decay_{arm}_s{s}"]["segs"]) for arm in ARMS}
     for s in SEEDS}


def c23a(t):
    return t["counters"] >= t["random"] + 2 * POINT, f"counters {t['counters']:.3f} random {t['random']:.3f}"


def c23b(t):
    edge, ceil = t["counters"] - t["random"], t["belady"] - t["random"]
    return edge >= 0.5 * ceil, f"edge {edge:.3f} vs belady edge {ceil:.3f}"


def c23c(t):
    best = max(t["lru"], t["lfu"])
    return t["counters"] >= best - 0.5 * POINT, f"counters {t['counters']:.3f} lru {t['lru']:.3f} lfu {t['lfu']:.3f}"


def c23d(t):
    return t["counters_u"] >= t["random_u"] + 2 * POINT, \
        f"uniform: counters {t['counters_u']:.3f} random {t['random_u']:.3f}"


CLAIMS = {
    "C23a counters beat random eviction (E1)": c23a,
    "C23b counters capture half of Belady's edge": c23b,
    "C23c combined price not beaten by its parts": c23c,
    "C23d counters beat random without structure (predicted-fail)": c23d,
}

for s in SEEDS:
    print(f"seed {s}: " + "  ".join(f"{arm} {A[s][arm]:.3f}" for arm in ARMS))
save("decay", {name: verdict(name, [test(A[s]) for s in SEEDS]) for name, test in CLAIMS.items()})
