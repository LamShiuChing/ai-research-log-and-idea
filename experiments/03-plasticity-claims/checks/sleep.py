"""C15a-C15i (CLAIMS.md, sleep section): 20-cycle wake/sleep consolidation on sleep_run.py.
TREND_BAR is derived from the registered formula max(0.05 pt, 2*sigma_null) on the seed-0 null
pilot — the formula is frozen at registration; the pilot file is regenerable."""
import json
import statistics

from lib import RESULTS, SEEDS, results, save, verdict

P0, FOUND, RVISITS, POINT = 0, (1, 2, 3, 4), (1, 2, 4, 8, 16), 0.01

BASE = ["--reservoir", 8000, "--sleep-old", 48]  # pilot-frozen constants (CLAIMS.md pilot record)
ARMS = dict(stored=["--arm", "stored"], labels=["--arm", "labels"], selfgen=["--arm", "selfgen"],
            weightadd=["--arm", "weightadd"], rebirth=["--arm", "rebirth"], null=["--arm", "null"],
            ensemble=["--arm", "ensemble"], scratch=["--arm", "scratch"],
            oracle=["--reservoir", 0],  # appended after BASE: the later flag wins
            prob_short=["--examples", 2500], prob_long=["--examples", 40000],
            accel=["--cycles", 60], conflict=["--conflict-cycle", 10])


def meanF(row):
    return sum(row[f] for f in FOUND) / len(FOUND)


def s_ret(r):
    return meanF(r["acc"][0]) - meanF(r["acc"][-1])


def s_new(r):
    drops = [r["comp_test"][c - 1] - r["acc"][c][d] for c, d in enumerate(r["sched"], 1)
             if d >= 6 and c != r["args"]["conflict_cycle"]]
    return sum(drops) / len(drops)


def s_trend(r, lo, hi):
    a = r["acc"]
    return sum(meanF(a[c - 1]) - meanF(a[c]) for c in range(lo, hi + 1)) / (hi - lo + 1)


def s_plast(r):
    fresh = [c for c, d in enumerate(r["sched"], 1) if d >= 6]
    return r["online"][fresh[0] - 1] - r["online"][fresh[-1] - 1]


def sigma_null(r):
    return statistics.stdev(meanF(r["acc"][c]) - meanF(r["acc"][c - 1]) for c in range(1, len(r["acc"])))


specs = {f"sleep_{arm}_s{s}": [*BASE, *args, "--seed", s] for s in SEEDS for arm, args in ARMS.items()}
specs["sleep_null_s0"] = [*BASE, "--arm", "null", "--seed", 0]
res = results(specs, script="sleep_run.py")
runs = {s: {arm: res[f"sleep_{arm}_s{s}"] for arm in ARMS} for s in SEEDS}
TREND_BAR = max(0.05 * POINT, 2 * sigma_null(res["sleep_null_s0"]))
print(f"TREND_BAR {TREND_BAR:.5f} (2*sigma_null(seed 0) = {2 * sigma_null(res['sleep_null_s0']):.5f})")


def c15a(r):
    ret, new = s_ret(r["stored"]), s_new(r["stored"])
    return ret <= 2 * POINT and new <= 2 * POINT, f"stored S_ret {ret:.3f} S_new {new:.3f}"


def c15b(r):
    t20, t60 = s_trend(r["stored"], 11, 20), s_trend(r["accel"], 41, 60)
    return t20 <= TREND_BAR and t60 <= TREND_BAR, f"trend 11-20 {t20:.5f}  accel 41-60 {t60:.5f}"


def c15c(r):
    p = s_plast(r["stored"])
    return p <= POINT, f"S_plast {p:.3f}"


def c15d(r):
    ret = s_ret(r["weightadd"])
    return ret > 2 * POINT, f"weightadd S_ret {ret:.3f}"


def c15e(r):
    ret, new = s_ret(r["selfgen"]), s_new(r["selfgen"])
    return ret > 2 * POINT or new > 2 * POINT, f"selfgen S_ret {ret:.3f} S_new {new:.3f}"


def c15f(r):
    ret, new = s_ret(r["rebirth"]), s_new(r["rebirth"])
    return ret <= 2 * POINT and new <= 2 * POINT, f"rebirth S_ret {ret:.3f} S_new {new:.3f}"


def c15g(r):
    on = [r["stored"]["online"][c - 1] for c in RVISITS]
    dn = [r["stored"]["dnorm"][c - 1] for c in RVISITS]
    ok = all(o >= on[1] - 2 * POINT for o in on[2:]) and all(d <= 1.25 * dn[1] for d in dn[2:])
    return ok, f"R online {[f'{o:.3f}' for o in on]} dnorm {[f'{d:.1f}' for d in dn]}"


def c15h(r):
    cf = r["conflict"]["conflict_acc"][9] / r["conflict"]["comp_test"][9]
    args = r["conflict"]["args"]
    p = args["foundation"] / (4 * args["foundation"] + 9 * args["examples"])
    share = args["sleep_old"] * p / (args["sleep_old"] * p + args["batch"])  # old side's F1 gradient share
    return abs(cf - (1 - share)) <= 0.15, f"correction {cf:.3f} vs 1-alpha {1 - share:.3f}"


def c15i(r):
    drop = r["stored"]["acc"][0][P0] - r["stored"]["acc"][-1][P0]
    return drop <= 2 * POINT, f"P0 drop {drop:.3f}"


def c15j(r):
    gain = s_ret(r["stored"]) - s_ret(r["labels"])
    return gain < POINT, f"labels S_ret {s_ret(r['labels']):.3f} vs stored {s_ret(r['stored']):.3f}"


CLAIMS = {
    "C15a sleep consolidates without destroying": c15a,
    "C15b no compounding drift": c15b,
    "C15c plasticity survives sleeps": c15c,
    "C15d weight-addition merge fails retention": c15d,
    "C15e pure self-generated rehearsal fails": c15e,
    "C15f rebirth is safe in sleep": c15f,
    "C15g spawn-merge converges": c15g,
    "C15h merge dilutes corrections as predicted": c15h,
    "C15i never-rehearsed probe survives (predicted-fail)": c15i,
    "C15j true-label rehearsal buys nothing over self-logits": c15j,
}

for s in SEEDS:
    r = runs[s]
    print(f"seed {s}: null S_ret {s_ret(r['null']):.3f} sigma {sigma_null(r['null']):.5f}"
          f" | oracle S_ret {s_ret(r['oracle']):.3f} S_new {s_new(r['oracle']):.3f}"
          f" | ensemble F {meanF(r['ensemble']['acc'][-1]):.3f} P0 {r['ensemble']['acc'][-1][P0]:.3f}"
          f" | scratch mean {statistics.fmean(r['scratch']['acc'][-1]):.3f}"
          f" | prob 2.5k ret/new {s_ret(r['prob_short']):.3f}/{s_new(r['prob_short']):.3f}"
          f" 40k {s_ret(r['prob_long']):.3f}/{s_new(r['prob_long']):.3f}")
save("sleep", {name: verdict(name, [test(runs[s]) for s in SEEDS]) for name, test in CLAIMS.items()})
