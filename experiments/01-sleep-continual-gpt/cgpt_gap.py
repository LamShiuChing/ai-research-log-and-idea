"""Closing the gap: can stronger sleep bring the continual GPT from 1.99 to the all-at-once ceiling (1.81)?

Why (cgpt_sheet.py, 3 seeds): the dreams GPT learns each day as well as plain (1.83) and forgets 0.19 (plain 0.58),
ending at 1.99 against 1.81 for all six domains mixed. The rest of the gap is mostly forgetting that dreams leave.
Two knobs of the night (cgpt.night), everything else as cgpt.py (same 6 BabyLM days, checkpoint -> test -> warm start):
  weight   how strongly the night asks the model to keep its old predictions on its dreams (KL weight 1 -> 3)
  dreams   how many dreams the old self writes before the night (512 -> 2,048), for wider cover of what it knew
Arms: weight 1 / 512 dreams (cgpt.py's setting, rerun), weight 3, 2,048 dreams, weight 3 + 2,048 dreams. 3 seeds.

Predictions, stated before the first run (averages over 3 seeds):
  D1 the best arm ends at a final average <= 1.92 (closes >= 40% of the 0.18 gap).
  D2 that arm still learns each new day within 0.03 of plain (1.83).
  D3 weight 3 forgets < 0.12.

Results (results/cgpt_gap_first.log for the first two arms, results/cgpt_gap_2048.json/.log for the rest: the first
run's 2,048-dream arm died of GPU memory when a second job shared the GPU; 3 seeds each, ~50 minutes in all):
                          just learned   forgetting   final average
  weight 1, 512 dreams       1.833         0.204         2.003
  weight 3                   1.840         0.137         1.954
  2048 dreams                1.831         0.168         1.971
  weight 3, 2048 dreams      1.854         0.083         1.923      (plain 0.58 / 2.31; joint ceiling 1.81)
Verdicts:
- D1 failed by a hair: the best arm ends at 1.923 (line 1.92); it closes 42% of the 0.19 gap to the ceiling.
- D2 passed: it still learns each day within 0.03 of plain (1.854 vs 1.825-1.834).
- D3 failed: weight 3 alone forgets 0.137 (line 0.12).
Post hoc: the two knobs add up: stronger keeping x wider dreams cuts forgetting to 0.08 (14% of plain's) at a small
cost in learning (+0.02 bits per byte).
"""
import json
import statistics
import sys

import cgpt as C

ARMS = {"weight 1, 512 dreams": (1.0, 512), "weight 3": (3.0, 512), "2048 dreams": (1.0, 2048),
        "weight 3, 2048 dreams": (3.0, 2048)}
SEEDS = (0, 1, 2)


def main():
    smoke = "smoke" in sys.argv
    if smoke:
        C.STEPS = 20
    only = [a for a in sys.argv[1:] if a != "smoke"]
    data, results = C.load(), {}
    for arm, (weight, dreams) in ((k, v) for k, v in ARMS.items() if all(o in k for o in only)):
        C.DREAM_WEIGHT, C.DREAMS = weight, dreams
        print(arm, flush=True)
        summaries = [C.summary(C.run(data, f"gap {arm}", False, True, s)[0]) for s in ((0,) if smoke else SEEDS)]
        results[arm] = {k: {"mean": round(statistics.mean(s[k] for s in summaries), 4),
                            "seeds": [round(s[k], 4) for s in summaries]}
                        for k in ("just learned", "forgetting", "final average")}
        print("  " + json.dumps(results[arm]), flush=True)
    if not smoke:
        (C.ROOT / "results" / f"cgpt_gap{''.join('_' + o for o in only)}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
