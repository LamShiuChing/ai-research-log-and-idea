"""C33a-d (CLAIMS.md, childhood section). Verdict diet = 100M track, seeds 1-3; workers=1."""
import statistics

from lib import SEEDS, results, save, verdict

CRIT_MARGIN = 0.05  # C33b/d floor; 2x paired seed-sd checked at verdict against it

specs = {f"child_{arm}_s{s}": ["--track", "100M", "--arm", arm, "--seed", s]
         for arm in ("curated", "raw", "scratch_ladder") for s in SEEDS}
res = results(specs, workers=1, script="lm_child_run.py")


def last_trained(r, src):
    return max(i for i, s in enumerate(r["segs"]) if s == src)


def s_ret_child(r, src):
    j = r["probes"].index(src)
    k = last_trained(r, src)
    return r["bpc"][-1][j] - r["bpc"][k][j], len(r["bpc"]) - 1 - k


def end_mean(r):
    return statistics.fmean(r["bpc"][-1][:10])


def tok_to_crit(r, rung="rung1"):
    c = r["curves"][rung]
    for i in range(9, len(c)):
        if statistics.fmean(c[i - 9:i + 1]) <= r["crit"]:
            return (i + 1) * 4096
    return len(c) * 4096  # unreached: full budget as the conservative floor (registered)


def c33a(s):
    r = res[f"child_curated_s{s}"]
    ok = sum(d <= 0.006 * n for d, n in (s_ret_child(r, src) for src in r["probes"][:10]))
    return ok >= 8, f"{ok}/10 sources within 0.006/sleep"


def c33b(s):
    d = end_mean(res[f"child_raw_s{s}"]) - end_mean(res[f"child_curated_s{s}"])
    return d >= CRIT_MARGIN, f"raw - curated = {d:+.3f} (margin {CRIT_MARGIN})"


def c33c(s):
    t_child = tok_to_crit(res[f"child_curated_s{s}"])
    t_fresh = tok_to_crit(res[f"child_scratch_ladder_s{s}"])
    return t_child <= 0.5 * t_fresh, f"child {t_child} vs scratch {t_fresh} tokens to crit"


def c33d(s):
    j = res[f"child_raw_s{s}"]["probes"].index("aochildes")
    d = res[f"child_raw_s{s}"]["bpc"][-1][j] - res[f"child_curated_s{s}"]["bpc"][-1][j]
    return d >= CRIT_MARGIN, f"aochildes raw - curated = {d:+.3f}"


CLAIMS = {
    "C33a the sleep loop survives a real childhood": c33a,
    "C33b the curriculum pays (A10)": c33b,
    "C33c childhood buys the ladder (possession)": c33c,
    "C33d curated retains earliest childhood (predicted-fail)": c33d,
}

for s in SEEDS:
    print(f"seed {s}: curated end {end_mean(res[f'child_curated_s{s}']):.3f}"
          f" raw {end_mean(res[f'child_raw_s{s}']):.3f}")
save("childhood", {name: verdict(name, [test(s) for s in SEEDS]) for name, test in CLAIMS.items()})
