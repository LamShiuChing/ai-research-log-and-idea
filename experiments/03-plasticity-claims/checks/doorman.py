"""C16a-C16e (CLAIMS.md, doorman section): label-free routing on doorman_run.py."""
import statistics
from collections import Counter

from lib import SEEDS, results, save, verdict

POINT = 0.01
ARMS = dict(doorman=["--arm", "doorman"], oracle=["--arm", "oracle"], cont=["--arm", "continue"],
            spawn=["--arm", "spawn"], marginal=["--arm", "marginal"],
            doorman1k=["--arm", "doorman", "--seglen", 1000],
            oracle1k=["--arm", "oracle", "--seglen", 1000])

specs = {f"door_{arm}_s{s}": [*args, "--seed", s] for s in SEEDS for arm, args in ARMS.items()}
res = results(specs, script="doorman_run.py")
runs = {s: {arm: res[f"door_{arm}_s{s}"] for arm in ARMS} for s in SEEDS}


def online(r):
    return statistics.fmean(g["online"] for g in r["segs"] if g["rule"] != "noise")


def spawns(r, seg=None):
    return sum(e["action"] == "spawn" for g, evs in enumerate(r["events"])
               if seg is None or g == seg for e in evs)


def hard_realarm(r):
    """Routing events inside hard-rule segments after their first 50 batches."""
    hard = r["args"]["rules"] - 1
    n, start = 0, 0
    for g, evs in zip(r["segs"], r["events"]):
        if g["rule"] == hard:
            n += sum(e["step"] - start > 50 for e in evs)
        start += len(g["timeline"])
    return n


def c16a(r):
    gap = online(r["oracle"]) - online(r["doorman"])
    return gap <= 3 * POINT, f"oracle {online(r['oracle']):.3f} doorman {online(r['doorman']):.3f}"


def c16b(r):
    d, c, sp = online(r["doorman"]), online(r["cont"]), online(r["spawn"])
    ok = d >= c + 10 * POINT and d >= sp - 2 * POINT and spawns(r["doorman"]) <= spawns(r["spawn"]) / 2
    return ok, (f"doorman {d:.3f} cont {c:.3f} spawn {sp:.3f}"
                f"  spawns {spawns(r['doorman'])}/{spawns(r['spawn'])}")


def c16c(r):
    m = r["marginal"]
    fired = sum(len(evs) for evs in m["events"])
    return fired == 0 and abs(online(m) - online(r["cont"])) <= POINT, \
        f"marginal alarms {fired}  online {online(m):.3f} vs cont {online(r['cont']):.3f}"


def c16d(r):
    d = r["doorman"]
    noise = [g["seg"] for g in d["segs"] if g["rule"] == "noise"]
    worst = max(spawns(d, seg=g) for g in noise)
    hr = hard_realarm(d)
    return worst <= 2 and hr == 0, f"max spawns/noise-seg {worst}  hard re-alarms {hr}"


def c16e(r):
    gap = online(r["oracle1k"]) - online(r["doorman1k"])
    return gap <= 3 * POINT, f"oracle1k {online(r['oracle1k']):.3f} doorman1k {online(r['doorman1k']):.3f}"


def routing_accuracy(r):
    """Fraction of batches routed to the rule's own adapter, over recurrences of learned rules."""
    first, assign, hits, total = {}, {}, 0, 0
    for g in r["segs"]:
        if g["rule"] == "noise":
            continue
        if g["rule"] not in first:
            first[g["rule"]] = g
            half = g["timeline"][len(g["timeline"]) // 2:]
            assign[g["rule"]] = Counter(half).most_common(1)[0][0]
        else:
            hits += sum(t == assign[g["rule"]] for t in g["timeline"])
            total += len(g["timeline"])
    return hits / total if total else 0.0


CLAIMS = {
    "C16a doorman tracks the oracle": c16a,
    "C16b waking beats never-switching and always-spawning": c16b,
    "C16c stream is marginal-proof (validity)": c16c,
    "C16d noise and hard-but-known traps hold": c16d,
    "C16e short segments break the doorman (predicted-fail)": c16e,
}

for s in SEEDS:
    r = runs[s]
    print(f"seed {s}: routing {routing_accuracy(r['doorman']):.3f}  pool {r['doorman']['segs'][-1]['pool']}"
          f"  doorman {online(r['doorman']):.3f} oracle {online(r['oracle']):.3f}"
          f" cont {online(r['cont']):.3f} spawn {online(r['spawn']):.3f}")
save("doorman", {name: verdict(name, [test(runs[s]) for s in SEEDS]) for name, test in CLAIMS.items()})
