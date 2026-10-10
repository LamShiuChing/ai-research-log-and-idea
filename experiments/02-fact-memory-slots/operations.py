"""Are the new query-language operations exact? (`reverse`, `count`, `same`)

`follow` was measured at 1.000 end to end (ask.py). These three are the first extensions of the
query language, and the whole point of putting them in the logic specialist rather than in the
network is that they should be exact rather than merely accurate. So they are scored against the
ground-truth universe with **set equality**, not overlap: a reverse answer counts only if it is
exactly right, extra names and missing names both failing it.

`reverse` is the interesting one. A slot's address is a hash of (relation, subject), so a
question that fixes the *object* has no address to compute. Rather than build an inverse index —
an external fact table, which THEORY.md §8 rules out — it asks the ordinary forward question of
every entity at once, in one batched forward pass, and keeps the ones that point at the target.
That keeps every fact in network parameters and costs O(entities), which is the ceiling to watch.

    python operations.py            # ~1 min
    python operations.py 300        # targets per operation
"""

import sys
import time
from random import Random

import torch

from glt.corpus import build
from glt.model import GPTConfig
from glt.universe import generate
from glt.workspace import count, follow, reverse, same, solve

sys.path.insert(0, ".")
from chat import Slots, load  # noqa: E402

torch.serialization.add_safe_globals([GPTConfig])


def incoming(universe):
    """(relation, object) -> the true set of subjects pointing at it. The answer key."""
    edges = {}
    for fact in universe.base:
        edges.setdefault((fact.path[0], fact.object), set()).add(fact.subject)
    return edges


def main(targets: int = 200, path: str = "out/online.pt") -> None:
    universe = generate(0, entities=200, relations=20, out_degree=10)
    corpus, model = build(universe), load(path)
    memory = Slots(model, corpus, universe.entities)
    edges, rng = incoming(universe), Random(0)
    asked = rng.sample(sorted(edges), targets)
    forward = {(f.path[0], f.subject): f.object for f in universe.base}

    start = time.time()
    answers = [solve(reverse(target, relation), memory)[1] for relation, target in asked]
    per_query = (time.time() - start) / targets * 1000

    exact = sum(set(note.answer) == edges[key] for note, key in zip(answers, asked))
    found = sum(len(set(note.answer) & edges[key]) for note, key in zip(answers, asked))
    extra = sum(len(set(note.answer) - edges[key]) for note, key in zip(answers, asked))
    true = sum(len(edges[key]) for key in asked)
    counted = sum(solve(count(reverse(target, relation)), memory)[1].answer == len(edges[key])
                  for key in asked for relation, target in [key])

    pairs = [(rng.choice(sorted(forward)), rng.choice(sorted(forward))) for _ in range(targets)]
    compared = sum(solve(same(follow(a[1], (a[0],)), follow(b[1], (b[0],))), memory)[1].answer
                   == (forward[a] == forward[b]) for a, b in pairs)

    print(f"reverse — {targets} targets, {true} true edges, one batched pass each")
    print(f"  exactly the right set   : {exact / targets:.3f}")
    print(f"  recall over all edges   : {found / true:.3f}")
    print(f"  names it invented       : {extra / true:.3f}")
    print(f"  cost                    : {per_query:.0f} ms per question ({len(universe.entities)} entities)")
    print(f"count   — right number    : {counted / targets:.3f}")
    print(f"same    — right verdict   : {compared / targets:.3f}   ({targets} random pairs)")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
