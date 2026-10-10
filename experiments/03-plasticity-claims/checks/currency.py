"""C22 (CLAIMS.md): retention pays, in prequential bits, iff the world recurs. Arms: bp vs replay,
on a non-recurring stream (10 fresh tasks) and a recurring one (pool of 10, 30 blocks)."""
from lib import SEEDS, results, save, verdict

FLAT = ["--tasks", 10, "--examples", 60000, "--lr", 0.05]
RECUR = ["--tasks", 30, "--examples", 20000, "--lr", 0.05, "--pool", 10]
SPARSE = ["--tasks", 90, "--examples", 20000, "--lr", 0.05, "--pool", 30]  # C22b: one knob, 3x sparser
ARMS = dict(bp=[], replay=["--replay", 2000])
MARGIN = 0.10  # registered in CLAIMS.md before running

specs = {f"cur_{stream}_{arm}_s{s}": [*scale, *args, "--seed", s]
         for stream, scale in (("flat", FLAT), ("recur", RECUR), ("sparse", SPARSE))
         for arm, args in ARMS.items() for s in SEEDS}
res = results(specs)


def saving(stream, s):
    """Replay's fractional code-length saving over BP on one stream and seed."""
    bp, rp = (sum(res[f"cur_{stream}_{arm}_s{s}"]["bits"]) for arm in ("bp", "replay"))
    return (bp - rp) / bp


def c22(s):
    flat, recur = saving("flat", s), saving("recur", s)
    return flat < MARGIN and recur >= MARGIN, f"saving flat {flat:+.3f}  recur {recur:+.3f}"


def c22b(s):
    sparse = saving("sparse", s)
    return sparse < MARGIN, f"saving sparse {sparse:+.3f}"


for s in SEEDS:
    for stream in ("flat", "recur", "sparse"):
        for arm in ARMS:
            print(f"seed {s} {stream:6s} {arm:6s} total bits {sum(res[f'cur_{stream}_{arm}_s{s}']['bits']):,.0f}")
save("currency", {
    "C22 retention pays iff the world recurs": verdict(
        "C22 retention pays iff the world recurs", [c22(s) for s in SEEDS]),
    "C22b still no pay at 3x sparser recurrence": verdict(
        "C22b still no pay at 3x sparser recurrence", [c22b(s) for s in SEEDS]),
})
