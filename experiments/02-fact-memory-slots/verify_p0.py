"""P0 verify -- does the harness measure anything?

Runs two regimes over one or more universes:

- **closed book** (no context): the contamination check. Any score above chance means
  the backbone already knows the "novel" facts and every later number is worthless.
- **open book** (base facts in context): 1-hop recall, then multi-hop questions whose
  answers are entailed but never stated.
"""

import sys
import time

from glt.evaluate import Backbone, evaluate, pool
from glt.questions import questions, statements
from glt.universe import generate

REGIMES = ("closed book (contamination check)", "open book (facts in context)")


def main(seeds: tuple[int, ...] = (0,)) -> None:
    backbone = Backbone()
    runs: dict[str, list] = {label: [] for label in REGIMES}
    for seed in seeds:
        universe = generate(seed)
        qs = questions(universe)
        context = " ".join(statements(universe))
        print(f"\nseed={seed}: {len(universe.base)} base facts, {len(universe.derived)} entailed, {len(qs)} questions")
        for label, ctx in zip(REGIMES, ("", context)):
            start = time.time()
            runs[label].append(evaluate(backbone, qs, ctx))
            print(f"  {label} [{time.time() - start:.0f}s]\n{runs[label][-1]}")

    print(f"\n=== pooled over seeds {seeds} ===   chance = 25.0% forced-choice")
    for label, results in runs.items():
        print(f"\n{label}\n{pool(results)}")


if __name__ == "__main__":
    main(tuple(int(a) for a in sys.argv[1:]) or (0,))
