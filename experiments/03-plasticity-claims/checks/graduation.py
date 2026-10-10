"""C29a-C29b (CLAIMS.md, graduation section): the B3 noise-graduation gate on sleep_run.py."""
from lib import SEEDS, results, save, verdict

FOUND, POINT = (1, 2, 3, 4), 0.01
NOISE = ["--noise-cycles", "5,10,15", "--reservoir", 8000, "--sleep-old", 48]
HARD = [*NOISE, "--hard-cycle", 7]
ARMS = dict(gated=[*NOISE, "--graduate", 0.5], ungated=NOISE,
            progress=[*HARD, "--graduate", 0.025, "--graduate-mode", "progress"],
            level_hard=[*HARD, "--graduate", 0.5])

specs = {f"grad_{arm}_s{s}": [*args, "--seed", s] for s in SEEDS for arm, args in ARMS.items()}
res = results(specs, script="sleep_run.py")
runs = {s: {arm: res[f"grad_{arm}_s{s}"] for arm in ARMS} for s in SEEDS}

mF = lambda row: sum(row[f] for f in FOUND) / len(FOUND)
s_ret = lambda r: mF(r["acc"][0]) - mF(r["acc"][-1])


def c29a(r):
    g = r["gated"]["graduated"]
    noise_fail = all(not g[c - 1] for c in (5, 10, 15))
    real_pass = all(ok for c, ok in enumerate(g, 1) if c not in (5, 10, 15))
    return noise_fail and real_pass, f"noise rejected {noise_fail}  real passed {real_pass}"


def c29b(r):
    gap = s_ret(r["ungated"]) - s_ret(r["gated"])
    return abs(gap) <= POINT, f"ungated S_ret {s_ret(r['ungated']):.3f} gated {s_ret(r['gated']):.3f}"


def c29c(r):
    g = r["progress"]["graduated"]
    noise_fail = all(not g[c - 1] for c in (5, 10, 15))
    rest_pass = all(ok for c, ok in enumerate(g, 1) if c not in (5, 10, 15))
    lvl = r["level_hard"]["graduated"][6]
    return noise_fail and rest_pass, \
        f"noise rejected {noise_fail}  hard+real passed {rest_pass}  (level gate on hard: {'passed' if lvl else 'rejected'})"


CLAIMS = {
    "C29a noise never graduates; knowledge always does": c29a,
    "C29b merging noise is harmless (predicted-fail)": c29b,
    "C29c progress gate is world-free and strictly better": c29c,
}

for s in SEEDS:
    print(f"seed {s}: gated S_ret {s_ret(runs[s]['gated']):.3f}  ungated {s_ret(runs[s]['ungated']):.3f}"
          f"  graduated {sum(runs[s]['gated']['graduated'])}/20")
save("graduation", {name: verdict(name, [test(runs[s]) for s in SEEDS]) for name, test in CLAIMS.items()})
