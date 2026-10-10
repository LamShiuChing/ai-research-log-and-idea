"""C31 (CLAIMS.md, LM currency section): the C22-port keystone on the tier-2 substrate.
workers=1 — one 85M run owns the GPU at a time."""
from lib import SEEDS, results, save, verdict

specs = {f"lm_{arm}_s{s}": ["--arm", arm, "--seed", s] for arm in ("bp", "replay") for s in SEEDS}
res = results(specs, workers=1, script="lm_run.py")


def spikes(r):
    """Per-domain revisit spikes: visit-k first-decile bpc minus visit-(k-1) last-decile bpc."""
    return {d: [round(v[k][0] - v[k - 1][1], 3) for k in range(1, len(v))]
            for d, v in r["visits"].items()}


def c31(s):
    b, r = res[f"lm_bp_s{s}"]["bits"], res[f"lm_replay_s{s}"]["bits"]
    saving = 1 - r / b
    return saving >= 0.10, f"bits saving {saving:.1%} (bp {b:.0f} replay {r:.0f})"


for s in SEEDS:
    print(f"seed {s}: bp bpc {res[f'lm_bp_s{s}']['bpc']:.4f} replay {res[f'lm_replay_s{s}']['bpc']:.4f}"
          f"  bp spikes {spikes(res[f'lm_bp_s{s}'])}")
save("lm_currency", {"C31 retention pays >=10% in bits at LM scale":
                     verdict("C31 retention pays >=10% in bits at LM scale", [c31(s) for s in SEEDS])})
