"""C25a-C25c (CLAIMS.md, facts section): the expensive-relearning world on facts_run.py."""
from lib import SEEDS, results, save, verdict

POINT = 0.01
BASE = ["--n", 20000, "--s", 0.6]  # pilot-frozen world (CLAIMS.md pilot record)
ARMS = dict(bp=["--arm", "bp"], reservoir=["--arm", "reservoir"], random=["--arm", "random"],
            priced=["--arm", "priced"],
            random_u=["--arm", "random", "--s", 0], priced_u=["--arm", "priced", "--s", 0])

specs = {f"facts_{arm}_s{s}": [*BASE, *args, "--seed", s] for s in SEEDS for arm, args in ARMS.items()}
res = results(specs, script="facts_run.py")
R = {s: {arm: res[f"facts_{arm}_s{s}"] for arm in ARMS} for s in SEEDS}


def c25a(r):
    saving = 1 - r["reservoir"]["bits"] / r["bp"]["bits"]
    return saving >= 0.10, f"bits saving {saving:.1%} (bp {r['bp']['bits']:.0f})"


def c25b(r):
    d = r["priced"]["online"] - r["random"]["online"]
    return d >= 2 * POINT, f"priced {r['priced']['online']:.3f} random {r['random']['online']:.3f}"


def c25c(r):
    d = r["priced_u"]["online"] - r["random_u"]["online"]
    return d >= 2 * POINT, f"uniform: priced {r['priced_u']['online']:.3f} random {r['random_u']['online']:.3f}"


CLAIMS = {
    "C25a retention pays in bits under scarcity (C22 ported)": c25a,
    "C25b priced eviction beats random under scarcity (C23a ported)": c25b,
    "C25c priced beats random without structure (predicted-fail)": c25c,
}

for s in SEEDS:
    r = R[s]
    print(f"seed {s}: " + "  ".join(f"{arm} {r[arm]['online']:.3f}" for arm in ARMS)
          + f"  | first-visit {r['bp']['first_miss']:.3f} distinct {r['bp']['distinct']}")
save("facts", {name: verdict(name, [test(R[s]) for s in SEEDS]) for name, test in CLAIMS.items()})
