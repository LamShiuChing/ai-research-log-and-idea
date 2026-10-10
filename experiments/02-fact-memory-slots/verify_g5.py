"""G5 measurement -- retract a premise mid-context, then see what survives.

Not a P3 result: at P0 the "memory" is just the context window, so this establishes the
baseline a real memory tier has to beat, and proves the three-part measurement runs.
"""

import sys
import time

from glt.evaluate import Backbone
from glt.probes import pool, retract, score_retraction
from glt.universe import generate


def main(seed: int = 0, n: int = 6) -> None:
    universe = generate(seed)
    backbone = Backbone()
    scores = []
    for fact in universe.base[:n]:
        r = retract(universe, fact, seed)
        start = time.time()
        scores.append(score_retraction(backbone, universe, r))
        print(f"\n{r.fact.subject}: {r.old} -> {r.new}  "
              f"({len(r.cascade)} consequences, {time.time() - start:.0f}s)")
        print(scores[-1])

    print(f"\n=== pooled over {len(scores)} retractions (seed {seed}) ===")
    print(pool(scores))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0, int(sys.argv[2]) if len(sys.argv) > 2 else 6)
