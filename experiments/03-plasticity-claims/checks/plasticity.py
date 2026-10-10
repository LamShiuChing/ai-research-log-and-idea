"""C1-C6 (CLAIMS.md): loss of plasticity on online permuted MNIST. Runs five arms on the verdict
seeds (results/ is the cache) and prints one verdict per claim."""
from lib import SEEDS, results, save, verdict, windows

SCALE = ["--tasks", 200, "--examples", 10000]  # fixed by the seed-0 pilot, see CLAIMS.md
ARMS = dict(
    bp=["--lr", 0.05],
    adam=["--adam", "--lr", 1e-3],
    l2=["--lr", 0.05, "--wd", 1e-4],
    snp=["--lr", 0.05, "--wd", 1e-4, "--noise", 3e-5],
    snp_hi=["--lr", 0.05, "--wd", 1e-4, "--noise", 1e-4],
    cbp=["--lr", 0.05, "--cbp", 1e-4],
    norm=["--lr", 0.05, "--normreset"],
    rand=["--lr", 0.05, "--cbp", 1e-4, "--cbp-random"],
)
POINT = 0.01

res = results({f"plast_{arm}_s{s}": [*SCALE, *args, "--seed", s] for s in SEEDS for arm, args in ARMS.items()})
runs = {s: {arm: res[f"plast_{arm}_s{s}"] for arm in ARMS} for s in SEEDS}


def last(r):
    return windows(r["online"])[-1]


def drop(r):
    w = windows(r["online"])
    return max(w) - w[-1]


def trend(r, key):
    w = windows([d[key] for d in r["diag"]])
    return w[-1] - w[0]


def c1(r):
    d = drop(r["bp"])
    return d > POINT, f"bp best-last {d:.3f}"


def c2(r):
    a, b = last(r["adam"]), last(r["bp"])
    return a < b, f"adam {a:.3f} vs bp {b:.3f}"


def c3(r):
    a, b = last(r["l2"]), last(r["bp"])
    return a > b, f"l2 {a:.3f} vs bp {b:.3f}"


def c4(r):
    a, b, d = last(r["snp"]), last(r["bp"]), drop(r["snp"])
    return a > b and d <= POINT, f"snp {a:.3f} vs bp {b:.3f}, snp best-last {d:.3f}"


def c4b(r):
    a, b, d = last(r["snp_hi"]), last(r["bp"]), drop(r["snp_hi"])
    return a > b and d <= POINT, f"snp 1e-4 {a:.3f} vs bp {b:.3f}, best-last {d:.3f}"


def c5(r):
    a, b, d = last(r["cbp"]), last(r["bp"]), drop(r["cbp"])
    return a > b and d <= POINT, f"cbp {a:.3f} vs bp {b:.3f}, cbp best-last {d:.3f}"


def c6(r):
    dead, wmag, erank = (trend(r["bp"], k) for k in ("dead", "wmag", "erank"))
    return dead > 0 and wmag > 0 and erank < 0, f"bp dead {dead:+.3f}  |w| {wmag:+.4f}  erank {erank:+.1f}"


def c10(r):
    a, b, d = last(r["norm"]), last(r["bp"]), drop(r["norm"])
    return a > b and d <= POINT, f"norm-reset {a:.3f} vs bp {b:.3f}, norm-reset best-last {d:.3f}"


def c11(r):
    a, b = last(r["rand"]), last(r["cbp"])
    return a >= b - POINT, f"random-reinit {a:.3f} vs cbp {b:.3f}"


CLAIMS = {
    "C1 plain backprop loses plasticity": c1,
    "C2 Adam is worse than SGD": c2,
    "C3 L2 mitigates": c3,
    "C4 shrink-and-perturb nearly eliminates": c4,
    "C4b C4 at noise 1e-4": c4b,
    "C5 continual backprop maintains plasticity": c5,
    "C6 dead units and |w| rise, effective rank falls": c6,
    "C10 per-task weight-norm reset alone maintains plasticity": c10,
    "C11 random reinit matches CBP (utility adds nothing)": c11,
}

for s in SEEDS:
    for arm, r in runs[s].items():
        w = windows(r["online"])
        print(f"seed {s} {arm:5s} first {w[0]:.3f} best {max(w):.3f} last {w[-1]:.3f}")
save("plasticity", {name: verdict(name, [test(runs[s]) for s in SEEDS]) for name, test in CLAIMS.items()})
