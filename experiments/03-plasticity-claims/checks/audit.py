"""C17a-C17d (CLAIMS.md, audit section): ledger audit on audit_run.py. H is derived from the
registered formula 5*sd(canary-domain per-sleep drops) on the frognull seed-0 pilot; FROG_NOISE
was pilot-calibrated to erode the all-canary domains ~0.5-1.5 pt/sleep."""
import math
import statistics

from lib import SEEDS, results, save, verdict

P0, POINT = 0, 0.01
FROG_STEPS, FROG_LR = 1, 0.003  # pilot-frozen poison dose (~0.8 pt/sleep canary erosion)

ARMS = dict(ledger=["--arm", "ledger"], snapshot=["--arm", "snapshot"],
            frog=["--arm", "frog", "--cycles", 60, "--frog-steps", FROG_STEPS, "--frog-lr", FROG_LR],
            frognull=["--arm", "frognull", "--cycles", 60])

specs = {f"audit_{arm}_s{s}": [*args, "--seed", s] for s in SEEDS for arm, args in ARMS.items()}
specs["audit_frognull_s0"] = ["--arm", "frognull", "--cycles", 60, "--seed", 0]
res = results(specs, script="audit_run.py")
runs = {s: {arm: res[f"audit_{arm}_s{s}"] for arm in ARMS} for s in SEEDS}


def canary_deltas(r, lo):
    cs = r["canary_scores"]
    return [statistics.fmean(cs[c - 2]) - statistics.fmean(cs[c - 1]) for c in range(lo, len(cs) + 1)]


ONSET = res["audit_frognull_s0"]["args"]["frog_onset"]
H = 5 * statistics.stdev(canary_deltas(res["audit_frognull_s0"], ONSET))
print(f"H {H:.5f} (5*sd of frognull seed-0 canary-domain drops from onset)")


def correction(r, cycle):
    return r["conflict_acc"][cycle - 1] / r["comp_test"][9]


def rot(r, cycle):
    return math.ceil(r["audit"][cycle - 1]["k"] / r["args"]["m"])


def c17a(r):
    led = r["ledger"]
    cf = correction(led, 10)
    rolled = led["audit"][9]["rollback"]
    dmg, eps = led["args"]["damage_cycle"], led["args"]["eps"]
    pre, after = led["acc"][dmg - 1][2], max(led["acc"][dmg + 1][2], led["acc"][dmg + 2][2])
    ok = cf >= 0.72 and not rolled and after >= pre - eps
    return ok, f"correction {cf:.3f} rolled {rolled}  F2 {pre:.3f} -> {led['acc'][dmg][2]:.3f} -> {after:.3f}"


def c17b(r):
    cf = correction(r["snapshot"], 20)
    return cf >= 0.72, f"snapshot final correction {cf:.3f} (rolled@10 {r['snapshot']['audit'][9]['rollback']})"


def c17c(r):
    led = r["ledger"]
    d6 = [i for i, d in enumerate(led["slice_domains"]) if d == 6]
    first = {}
    for e in led["audit"]:
        if e["cycle"] > led["args"]["damage_cycle"]:  # natural packing-phase erosion doesn't count
            for i, dom in e["failed"]:
                if dom == 6:
                    first.setdefault(i, e["cycle"])
    R = rot(led, led["args"]["damage_cycle"])
    latest = max((first.get(i, 99) for i in d6), default=99)
    return latest - led["args"]["damage_cycle"] <= R, \
        f"dom-6 catches {[first.get(i) for i in d6]} (damage@{led['args']['damage_cycle']}, R={R})"


def ph_alarms(r):
    ph, k, out = 0.0, r["args"]["ph_k"], []
    for c, dt in zip(range(ONSET, len(r["canary_scores"]) + 1), canary_deltas(r, ONSET)):
        ph = max(0.0, ph + dt - k)
        if ph > H:
            out.append(c)
    return out


def c17d(r):
    frog, null = r["frog"], r["frognull"]
    R = rot(frog, len(frog["audit"]))
    fa, na = ph_alarms(frog), ph_alarms(null)
    first_item = next((e["cycle"] for e in frog["audit"] if e["cycle"] >= ONSET and e["failed"]), 99)
    null_item = sum(len(e["failed"]) for e in null["audit"] if e["cycle"] >= ONSET)
    ok = bool(fa) and fa[0] <= ONSET + 2 * R and fa[0] <= first_item and not na and null_item <= 1
    return ok, (f"PH alarm @{fa[0] if fa else None} (bar {ONSET + 2 * R})  first item-fire @{first_item}  "
                f"null: PH alarms {len(na)} item-fails {null_item}")


CLAIMS = {
    "C17a ledger audit accepts corrections; covered damage self-repairs": c17a,
    "C17b snapshot-anchored audit also accepts the correction (predicted-fail)": c17b,
    "C17c rotation catches unhealed damage within R sleeps": c17c,
    "C17d boiled frog caught by aggregate only": c17d,
}

for s in SEEDS:
    led = runs[s]["ledger"]
    mF = lambda row: sum(row[f] for f in (2, 3, 4)) / 3  # F1 is corrected mid-run; old-label probe excluded
    print(f"seed {s}: ledger S_ret(F2-4) {mF(led['acc'][0]) - mF(led['acc'][-1]):.3f}"
          f"  P0 drop {led['acc'][0][P0] - led['acc'][-1][P0]:.3f}"
          f"  rollbacks {sum(e['rollback'] for e in led['audit'])}"
          f"  snapshot rollbacks {sum(e['rollback'] for e in runs[s]['snapshot']['audit'])}")
save("audit", {name: verdict(name, [test(runs[s]) for s in SEEDS]) for name, test in CLAIMS.items()})
