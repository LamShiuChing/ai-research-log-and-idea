"""C13a-c + C30a-c (CLAIMS.md, coverage & bounded buffer section): round 12.
H* is derived in-code from the seed-0 bp pilot by the registered formula (first >=200-presentation
log2 bin whose recall falls halfway to chance); the pilot file is regenerable."""
import math
import statistics

from lib import SEEDS, results, save, verdict

POINT = 0.01
NS = (10000, 20000, 40000, 80000)
WORLD = ["--n", 20000, "--s", 0.6]  # frozen C25 world (total 300000, slots 8000 are defaults)
FOUND = (1, 2, 3, 4)  # sleep instrument constants, verbatim from checks/sleep.py
SLEEP8K = ["--reservoir", 8000, "--sleep-old", 48, "--arm", "stored"]


def walk_name(kind, n, s):
    if kind == "b" and n == 20000:
        return f"facts_random_s{s}"  # identical spec to C25's bounded run: reuse it
    return f"walk_{kind}{n // 1000}k_s{s}"


specs = {walk_name("b", n, s): ["--n", n, "--s", 0.6, "--total", 15 * n, "--slots", 8000,
                                "--arm", "random", "--seed", s] for n in NS for s in SEEDS}
specs |= {walk_name("u", n, s): ["--n", n, "--s", 0.6, "--total", 15 * n, "--slots", n,
                                 "--arm", "random", "--seed", s] for n in NS for s in SEEDS}
specs |= {f"drift_{arm}_s{s}": [*WORLD, "--rho", 0.1, "--arm", arm, "--seed", s]
          for arm in ("bp", "naive", "aware") for s in SEEDS}
specs["hstar_bp_s0"] = [*WORLD, "--arm", "bp", "--seed", 0]
res = results(specs, script="facts_run.py")

sleep_specs = {f"cov_sleep2000_s{s}": ["--reservoir", 2000, "--sleep-old", 48, "--arm", "stored",
                                       "--seed", s] for s in SEEDS}
sleep_specs |= {f"sleep_stored_s{s}": [*SLEEP8K, "--seed", s] for s in SEEDS}  # existing verdicts
sleep = results(sleep_specs, script="sleep_run.py")


def h_star(r):
    edges = [(2 ** b, c / n) for b, (n, c) in enumerate(zip(r["gap_n"], r["gap_c"])) if n >= 200]
    r0 = edges[0][1]
    return next(e for e, rec in edges if rec <= (r0 + 0.10) / 2)


H_STAR = h_star(res["hstar_bp_s0"])
print(f"H* = {H_STAR}")


def lsq(zs, ys):
    """Least squares y = a + b*z; returns sse."""
    n, mz, my = len(zs), statistics.fmean(zs), statistics.fmean(ys)
    szz = sum((z - mz) ** 2 for z in zs)
    b = sum((z - mz) * (y - my) for z, y in zip(zs, ys)) / szz if szz else 0.0
    return sum((y - (my - b * mz) - b * z) ** 2 for z, y in zip(zs, ys))


def hinge_fit(xs, ys):
    """max(a, a + b(x - x0)) == a + b*max(0, x - x0): grid x0, ls in (a, b). Returns (sse, x0)."""
    return min((lsq([max(0.0, x - x0) for x in xs], ys), x0)
               for x0 in (12 + i * 0.02 for i in range(251)))


def gaps(s):
    return [res[walk_name("u", n, s)]["online"] - res[walk_name("b", n, s)]["online"] for n in NS]


def s_drift(arm, s, cond):
    post = res[f"drift_{arm}_s{s}"]["drift"]["post"]
    n = sum(post[str(i)][0] for i in cond)
    return sum(post[str(i)][1] for i in cond) / n


def phi(s):
    r = res[f"drift_aware_s{s}"]
    k = H_STAR.bit_length() - 1
    cols = [p[k] for p, t in zip(r["phi"], r["phi_t"]) if t <= r["args"]["total"] - H_STAR]
    return statistics.fmean(cols)


def s_ret(r):
    a = r["acc"]
    return statistics.fmean(a[0][f] for f in FOUND) - statistics.fmean(a[-1][f] for f in FOUND)


def s_trend(r, lo, hi):
    a = r["acc"]
    return sum(statistics.fmean(a[c - 1][f] for f in FOUND) - statistics.fmean(a[c][f] for f in FOUND)
               for c in range(lo, hi + 1)) / (hi - lo + 1)


def c13a(s):
    xs, ys = [math.log2(n) for n in NS], gaps(s)
    sse_l, (sse_h, x0) = lsq(xs, ys), hinge_fit(xs, ys)
    ok = sse_h <= 0.5 * sse_l and 8000 <= 2 ** x0 <= 32000
    return ok, (f"gaps {[f'{g:.3f}' for g in ys]} sse line {sse_l:.5f} hinge {sse_h:.5f}"
                f" knee {2 ** x0:.0f}")


def c13b(s):
    r2, r8 = s_ret(sleep[f"cov_sleep2000_s{s}"]), s_ret(sleep[f"sleep_stored_s{s}"])
    return r2 >= 2 * r8 + POINT, f"S_ret 2000 {r2:.3f} vs 8000 {r8:.3f} (bar {2 * r8 + POINT:.3f})"


def c13c(s):
    t2 = s_trend(sleep[f"cov_sleep2000_s{s}"], 11, 20)
    t8 = s_trend(sleep[f"sleep_stored_s{s}"], 11, 20)
    return t2 >= 2 * t8 + 0.0003, f"trend 2000 {t2:.5f} vs 8000 {t8:.5f} (bar {2 * t8 + 0.0003:.5f})"


def c30a(s):
    p = phi(s)
    return p >= 0.5, f"phi_store(H*={H_STAR}) = {p:.3f}"


def c30b(s):
    cond = res[f"drift_naive_s{s}"]["drift"]["resident_at_event"]
    d = s_drift("aware", s, cond) - s_drift("bp", s, cond)
    return d >= -2 * POINT, f"aware-bp {d:+.3f} on {len(cond)} at-risk facts"


def c30c(s):
    cond = res[f"drift_naive_s{s}"]["drift"]["resident_at_event"]
    d = s_drift("naive", s, cond) - s_drift("bp", s, cond)
    return d >= -2 * POINT, f"naive-bp {d:+.3f} on {len(cond)} at-risk facts"


CLAIMS = {
    "C13a coverage starves as a threshold, not a slope": c13a,
    "C13b the 2000-reservoir glimpse survives fresh seeds": c13b,
    "C13c the novelty tax inflates when coverage starves": c13c,
    "C30a the stream re-tests what the store keeps (phi >= 0.5)": c30a,
    "C30b contradiction-awareness closes drift to within noise of bp": c30b,
    "C30c stale rehearsal is harmless (predicted-fail)": c30c,
}

for s in SEEDS:
    dm = res[f"drift_bp_s{s}"]["drift"]["mass"]
    dw = res[f"drift_aware_s{s}"]["drift"]["dwell"]
    cond = res[f"drift_naive_s{s}"]["drift"]["resident_at_event"]
    print(f"seed {s}: drift mass {dm:.3f} (gate .05-.15)  dwell median {dw[len(dw) // 2] if dw else '-'}"
          f" corrected {len(dw)}/{len(cond)}  | walk gaps {[f'{g:.3f}' for g in gaps(s)]}"
          f" | S_drift bp {s_drift('bp', s, cond):.3f} naive {s_drift('naive', s, cond):.3f}"
          f" aware {s_drift('aware', s, cond):.3f}")
save("coverage", {name: verdict(name, [test(s) for s in SEEDS]) for name, test in CLAIMS.items()})
