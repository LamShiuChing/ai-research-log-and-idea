"""C35a-e (CLAIMS.md, Stage C section): the first flywheel turn. Margins: max(0.02, 2x seed-sd
of each claim's own delta), printed here and recorded in CLAIMS at verdict; workers=1 (85M)."""
import statistics

from lib import SEEDS, results, save, verdict

specs = {f"fw_{arm}_s{s}": ["--arm", arm, "--seed", s]
         for arm in ("self", "rule", "raw") for s in SEEDS}
res = results(specs, workers=1, script="flywheel_run.py")

DEPTHS = ("1", "2", "3")


def overall(r, cycle):
    return statistics.fmean(r["acc"][cycle][d] for d in DEPTHS)


def lang(r, cycle):
    return statistics.fmean(r["bpc"][cycle][:10])


def margin(deltas):
    return max(0.02, 2 * statistics.stdev(deltas.values()))


d_gain = {s: overall(res[f"fw_self_s{s}"], -1) - overall(res[f"fw_self_s{s}"], 0) for s in SEEDS}
d_rule = {s: overall(res[f"fw_self_s{s}"], -1) - overall(res[f"fw_rule_s{s}"], -1) for s in SEEDS}
d_raw = {s: overall(res[f"fw_self_s{s}"], -1) - overall(res[f"fw_raw_s{s}"], -1) for s in SEEDS}
d_tool = {s: res[f"fw_self_s{s}"]["tool_final"] - res[f"fw_self_s{s}"]["acc"][-1]["3"] for s in SEEDS}
d_lang = {s: lang(res[f"fw_self_s{s}"], -1) - lang(res[f"fw_self_s{s}"], 0) for s in SEEDS}
M_GAIN, M_RULE, M_RAW, M_TOOL = (margin(d) for d in (d_gain, d_rule, d_raw, d_tool))
print(f"margins: gain {M_GAIN:.3f} rule {M_RULE:.3f} raw {M_RAW:.3f} tool {M_TOOL:.3f}")

CLAIMS = {
    "C35a the flywheel turns (self-practice teaches)":
        lambda s: (d_gain[s] > M_GAIN, f"gain {d_gain[s]:+.3f} (bar {M_GAIN:.3f})"),
    "C35b self matches-or-beats the rule generator (de-hardcoding rung 4)":
        lambda s: (d_rule[s] >= -M_RULE, f"self-rule {d_rule[s]:+.3f} (margin {M_RULE:.3f})"),
    "C35c the language is kept (drift <= 0.10 bpc)":
        lambda s: (d_lang[s] <= 0.10, f"lang drift {d_lang[s]:+.3f} bpc"),
    "C35d raw matches graded (predicted-fail; C15e re-cut)":
        lambda s: (d_raw[s] <= M_RAW, f"self-raw {d_raw[s]:+.3f} (margin {M_RAW:.3f})"),
    "C35e the real tool pays (grounded beats direct at depth 3)":
        lambda s: (d_tool[s] > M_TOOL, f"tool-direct {d_tool[s]:+.3f} (bar {M_TOOL:.3f})"),
}

for s in SEEDS:
    r = res[f"fw_self_s{s}"]
    st = r["stats"][-1]
    reach = st["tool"] / max(1, st["attempts"])
    print(f"seed {s}: final acc {[round(r['acc'][-1][d], 3) for d in DEPTHS]}"
          f"  verified {st['verified']}/{st['attempts']}  tool-reach {reach:.3f}"
          f"  tool d3 {r['tool_final']:.3f}")
save("flywheel", {name: verdict(name, [test(s) for s in SEEDS]) for name, test in CLAIMS.items()})
