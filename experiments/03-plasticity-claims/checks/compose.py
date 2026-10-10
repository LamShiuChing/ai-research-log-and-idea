"""C21a-C21c (CLAIMS.md, composition section): the two-digit pair stream on compose_run.py."""
from lib import SEEDS, results, save, verdict

POINT = 0.01
ARMS = ("ft", "deep", "tap", "probe", "scratch")

specs = {f"comp_{arm}_s{s}": ["--arm", arm, "--seed", s] for s in SEEDS for arm in ARMS}
res = results(specs, script="compose_run.py")
T = {s: {arm: res[f"comp_{arm}_s{s}"]["test"] for arm in ARMS} for s in SEEDS}


def c21a(t):
    return t["ft"] - t["tap"] >= 5 * POINT, f"ft {t['ft']:.3f} tap {t['tap']:.3f}"


def c21b(t):
    return t["ft"] - t["deep"] <= 2 * POINT, f"ft {t['ft']:.3f} deep {t['deep']:.3f}"


def c21c(t):
    return t["scratch"] >= t["ft"] - 2 * POINT, f"scratch {t['scratch']:.3f} vs ft {t['ft']:.3f}"


CLAIMS = {
    "C21a composition gap: read-only tap trails direct FT": c21a,
    "C21b deep low-rank adapter closes the gap": c21b,
    "C21c primitives unnecessary at this budget (predicted-fail)": c21c,
}

for s in SEEDS:
    print(f"seed {s}: " + "  ".join(f"{arm} {T[s][arm]:.3f}" for arm in ARMS))
save("compose", {name: verdict(name, [test(T[s]) for s in SEEDS]) for name, test in CLAIMS.items()})
