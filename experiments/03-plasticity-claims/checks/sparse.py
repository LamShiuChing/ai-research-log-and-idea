"""SS-a/b/c (CLAIMS.md, SLEEP-sparse section): masked sleep vs the house sleep on the v3 child.
Statistics per run (sparse_run.py json): S_lang = final - pre childhood b/char (+ = forgot);
S_stage = mean stage-held drift (b/tok); world knowledge of a slice = the never-consolidated
floor slice's bits minus the slice's bits at the same probe, minus that difference at the
pre-probe (format learning and drift move every slice alike; item difficulty differs per slice); acquisition = mean over slices of that knowledge right after the slice's
own sleep (cloze, + = learned); composition = the same on the 2-hop chain bits; P0 erosion =
slice 0's knowledge right after its sleep minus at the end (the never-rehearsed slice).
Margins are pilot-frozen (seed-0 record in CLAIMS): MARGIN = the larger of 2x the full arm's
test-retest sd (seeds 0/100) and the floor, per statistic."""
import statistics

from lib import SEEDS, results, save, verdict

# frozen 2026-09-03 12:12 from the same-world full pair (sp_pilot_full_s0 / _s100w0): 2x the
# paired difference, floored at lang 0.02 / stage 0.02 / acq 0.5 / comp 0.5 / p0 0.25
MARGIN = dict(lang=0.220, stage=0.02, acq=0.5, comp=3.06, p0=0.563)
GAP_SUM = sum(MARGIN[k] for k in ("lang", "stage", "acq", "comp"))

def arm(name, *extra):
    return {f"sp_{name}_s{s}": ["--sleep", name.split("_")[0].rstrip("05"), *extra, "--settle", 150, "--seed", s]
            for s in SEEDS}


# dict order = run order (workers=1): the product path first (SS-a/b, SS-e1/e2), then the mechanism
# controls (SS-c, tag control) and the tag control at the budget that binds (pilot: random matches
# topk at 0.02, trails by 3.9 at 0.005)
specs = (arm("full") | arm("topk") | arm("direct") | arm("direct_n32", "--naps", 32)
         | arm("random") | arm("twotier") | arm("topk005", "--frac", 0.005) | arm("random005", "--frac", 0.005))


def stats(r):
    h = r["hist"]
    pre, fin = h[0], h[-1]
    n = len(h) - 1

    def know(t, c, key):
        """Slice c's knowledge at probe t: its gap to the floor slice (index n), minus the same
        gap at the pre-probe (slices differ in item difficulty by ~3 bits before any sleep)."""
        gap = lambda row: row[n][key] - row[c][key]
        return gap(h[t]["world"]) - gap(pre["world"])

    return dict(
        lang=fin["lang"] - pre["lang"],
        stage=statistics.fmean(fin["stage"][k] - pre["stage"][k] for k in pre["stage"]),
        acq=statistics.fmean(know(c + 1, c, "cloze_bits") for c in range(n)),
        comp=statistics.fmean(know(c + 1, c, "comp_bits") for c in range(n)),
        p0=know(1, 0, "cloze_bits") - know(n, 0, "cloze_bits"),
    )


def trail(lead, arm, s):
    """Per statistic, how far `arm` trails `lead` (+ = worse) at seed s."""
    f, m = stats(res[f"sp_{lead}_s{s}"]), stats(res[f"sp_{arm}_s{s}"])
    return dict(lang=m["lang"] - f["lang"], stage=m["stage"] - f["stage"],
                acq=f["acq"] - m["acq"], comp=f["comp"] - m["comp"], p0=m["p0"] - f["p0"])


def gap(arm, s):
    return trail("full", arm, s)


def gap_sum(lead, arm, s):
    return sum(trail(lead, arm, s)[k] for k in ("lang", "stage", "acq", "comp"))


def ss_a(s):
    g = gap("topk", s)
    broke = [k for k in ("lang", "stage", "acq", "comp") if g[k] > MARGIN[k]]
    return not broke, f"topk-full gaps {({k: round(v, 3) for k, v in g.items()})} broke {broke}"


def ss_b(s):
    g = gap("topk", s)["p0"]
    return g < -MARGIN["p0"], f"P0 erosion topk-full {g:.3f} bits"


def ss_c(s):
    d = gap_sum("twotier", "topk", s)
    return d > GAP_SUM, f"twotier beats topk by {d:.3f} on the gap sum (bar {GAP_SUM:.2f})"


def ss_e1(s):
    """Teacherless suffices and out-learns the tag-distill arm."""
    g, more = gap("direct", s), -trail("topk", "direct", s)["acq"]
    broke = [k for k in ("lang", "stage") if g[k] > MARGIN[k]]
    return not broke and more > MARGIN["acq"], (f"direct-full gaps {({k: round(v, 3) for k, v in g.items()})} broke {broke};"
                                              f" direct beats topk on acq by {more:.2f} bits (bar {MARGIN['acq']})")


def ss_e2(s):
    """Nap-per-action: direct at 32 naps/round within margin of direct at 1."""
    t = trail("direct", "direct_n32", s)
    broke = [k for k in ("lang", "stage", "acq", "comp") if t[k] > MARGIN[k]]
    return not broke, f"direct N=32 trails N=1 by {({k: round(v, 3) for k, v in t.items()})} broke {broke}"


def tag_control(s, frac=""):
    d = gap_sum(f"topk{frac}", f"random{frac}", s)
    return d > GAP_SUM, f"frac {frac or '0.02'}: random-mask trails topk by {d:.3f} on the gap sum (bar {GAP_SUM:.2f})"


CLAIMS = {
    "SS-a masked sleep suffices (retention + acquisition + composition within margin of full)": ss_a,
    "SS-b structural retention pays (masked erodes the never-rehearsed slice less)": ss_b,
    "SS-e1 teacherless suffices and out-learns tag-distill (retention within margin of full; acq > topk by margin)": ss_e1,
    "SS-e2 nap-per-action: teacherless at 32 naps/round within margin of 1 nap/round on all four": ss_e2,
    "SS-c two-tier beats plain top-k (predicted-fail candidate)": ss_c,
    "tag control at frac 0.02: the tag beats a random mask of equal size": tag_control,
    "tag control at frac 0.005 (the budget that binds): the tag beats a random mask": lambda s: tag_control(s, "005"),
}

if __name__ == "__main__":
    res = results(specs, workers=1, script="sparse_run.py")
    for name in specs:
        print(f"{name}: {({k: round(v, 3) for k, v in stats(res[name]).items()})}")
    save("sparse", {name: verdict(name, [test(s) for s in SEEDS]) for name, test in CLAIMS.items()})
