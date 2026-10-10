"""C32a-c (CLAIMS.md, LM sleep section): the C15-port on the tier-2 substrate.
TREND_BAR derives from the seed-0 null pilot by the registered formula; workers=1 (85M runs)."""
import statistics

from lib import SEEDS, results, save, verdict

FOUND = slice(1, 5)  # DOMS order: P0, F1-4, R, fresh...
S_RET_BAR = 0.10  # bpc; registered convention

specs = {f"lms_stored_s{s}": ["--arm", "stored", "--cycles", 20, "--seed", s] for s in SEEDS}
specs |= {f"lms_weightadd_s{s}": ["--arm", "weightadd", "--seed", s] for s in SEEDS}
specs["lms_null_s0"] = ["--arm", "null", "--seed", 0]  # bars come from seed 0 only (registered)
res = results(specs, workers=1, script="lm_sleep_run.py")


def fmean(row):
    return statistics.fmean(row[FOUND])


def s_ret(r):
    return fmean(r["bpc"][-1]) - fmean(r["bpc"][0])


def trend(r, lo, hi):
    return statistics.fmean(fmean(r["bpc"][c]) - fmean(r["bpc"][c - 1]) for c in range(lo, hi + 1))


sigma_null = statistics.stdev(fmean(res["lms_null_s0"]["bpc"][c]) - fmean(res["lms_null_s0"]["bpc"][c - 1])
                              for c in range(1, len(res["lms_null_s0"]["bpc"])))
TREND_BAR = max(0.005, 2 * sigma_null)
print(f"TREND_BAR {TREND_BAR:.4f} (2*sigma_null = {2 * sigma_null:.4f})")


def c32a(s):
    v = s_ret(res[f"lms_stored_s{s}"])
    return v <= S_RET_BAR, f"stored S_ret {v:.3f} bpc"


def c32b(s):
    v = trend(res[f"lms_stored_s{s}"], 13, 20)
    return v <= TREND_BAR, f"recycled trend {v:.4f} bpc/cycle"


def c32c(s):
    v = s_ret(res[f"lms_weightadd_s{s}"])
    return v <= 2 * S_RET_BAR, f"weightadd S_ret {v:.3f} bpc"


CLAIMS = {
    "C32a sleep consolidates without destroying (language)": c32a,
    "C32b novelty tax is per-novelty, not per-sleep (language)": c32b,
    "C32c weight-addition merge retains (predicted-fail; industry LoRA-merge)": c32c,
}

for s in SEEDS:
    r = res[f"lms_stored_s{s}"]
    print(f"seed {s}: stored S_ret {s_ret(r):.3f} P0 drop {r['bpc'][-1][0] - r['bpc'][0][0]:.3f}"
          f" weightadd S_ret {s_ret(res[f'lms_weightadd_s{s}']):.3f}")
save("lm_sleep", {name: verdict(name, [test(s) for s in SEEDS]) for name, test in CLAIMS.items()})
