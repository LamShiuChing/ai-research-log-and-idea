"""Step 5 of HANDOFF.md §4 -- G5a: does deleting a premise withdraw its consequences?

Session 2 could not ask this. G5a needs consequences that were *actually derived* before the
correction, and Llama composed at 0.15, so only 2 of 48 were eligible: you cannot observe the
withdrawal of a conclusion the model never reached. The workspace loop derives them at 1.000
(ask.py), so every consequence is now eligible and the question is finally measurable.

Two claims are separated here, because they can fail independently:

- **bookkeeping** -- the premises a note records must be the ones the entailment graph says.
  If provenance is wrong, the cascade deletes the wrong things, exactly and confidently.
- **substrate** -- with the fact zeroed out of the slots and every board thrown away, its
  consequences must stop being derivable, and nothing else may move. This is the claim
  THEORY.md §3 doubted for weights and §8 expects from bookkeeping.

The second is the real test: it re-derives from scratch rather than editing the board, so a
consequence that survives is genuine continued influence -- the failure mode of §7, where a
retracted belief keeps acting through what it left behind.

    python cascade.py                 # ~8 min on a 4080
    python cascade.py 24 200          # premises retracted, controls per premise
"""

import sys
from random import Random

import torch

from glt.corpus import build
from glt.model import GPTConfig
from glt.questions import cooccurring
from glt.universe import generate
from glt.workspace import follow, solve
from perturb import online_facts

sys.path.insert(0, ".")
from chat import Slots, address, load, slots_of  # noqa: E402

torch.serialization.add_safe_globals([GPTConfig])


class Cached(Slots):
    """The memory specialist with one forward pass per distinct hop, dropped when slots change."""

    def __init__(self, *args):
        super().__init__(*args)
        self.memo = {}

    def follow(self, relation, subject):
        if (relation, subject) not in self.memo:
            self.memo[relation, subject] = super().follow(relation, subject)
        return self.memo[relation, subject]


def resolve(derivations, memory):
    """Answer each question from a *fresh* board, so no note survives a retraction."""
    return [solve(follow(d.fact.subject, d.fact.path), memory)[1] for d in derivations]


def said(note):
    return note.answer if note is not None else None


def eligible(universe, held):
    """Consequences of held-out facts, minus any whose answer is guessable from the subject alone."""
    adjacent = cooccurring(universe)
    return [d for d in universe.derived
            if not d.premises.isdisjoint(held) and d.fact.object not in adjacent[d.fact.subject]]


def zeroing(model, corpus, memory):
    """Retract by emptying the slot row. Cached answers go with it, or the board reads a stale hop."""
    def retract(fact):
        slot = address(model, corpus, fact.path[0], fact.subject)
        saved = slots_of(model).values[slot].clone()

        def put(row):
            with torch.no_grad():
                slots_of(model).values[slot] = row
            memory.memo.clear()

        put(0)
        return lambda: put(saved)
    return retract


def measure(derivations, held, memory, retract, premises: int = 24, controls: int = 200) -> None:
    """The G5a suite against any store: `retract(fact)` empties it and returns how to put it back.

    The board, the provenance and the arithmetic are the store's only contract, so running this
    on a second substrate is what turns "the upper stack is store-independent" into a number.
    """
    base = resolve(derivations, memory)
    answered = sum(said(note) == d.fact.object for note, d in zip(base, derivations))
    faithful = sum(note is not None and set(note.grounds) == set(d.premises)
                   for note, d in zip(base, derivations))
    print(f"consequences of slot-written facts: {len(derivations)}")
    print(f"  derived correctly before any retraction : {answered / len(derivations):.3f}")
    print(f"  premises recorded == entailment graph   : {faithful / len(derivations):.3f}")

    rng = Random(0)
    pool = sorted({p for d in derivations for p in d.premises} & held, key=lambda f: (f.subject, f.path))
    targets = rng.sample(pool, premises)
    withdrawn, collateral, recalled, hit, missed, survived, rerouted = [], [], [], 0, 0, 0, 0

    for target in targets:
        restore = retract(target)

        children = [i for i, d in enumerate(derivations) if target in d.premises]
        others = rng.sample([i for i, d in enumerate(derivations) if target not in d.premises], controls)
        after = dict(zip(children + others, resolve([derivations[i] for i in children + others], memory)))
        changed = {i for i, note in after.items() if said(note) != said(base[i])}
        predicted = {i for i in after if base[i] is not None and target in base[i].grounds}

        recalled.append(memory.follow(target.path[0], target.subject)[0] == target.object)
        withdrawn.append(len(changed & set(children)) / len(children))
        collateral.append(len(changed & set(others)) / len(others))
        hit += len(predicted & changed)
        missed += len(predicted ^ changed)
        # A consequence whose answer did not move is only continued influence if it reached that
        # answer the same way. A different intermediate landing on the same name is coincidence.
        kept = set(children) - changed
        survived += len(kept)
        rerouted += sum(after[i].grounds[0].object != base[i].grounds[0].object for i in kept)

        restore()

    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    print(f"\nretracted {premises} premises, one at a time, {controls} controls each")
    print(f"  the premise itself still recalled       : {mean(recalled):.3f}   (G5, target retraction)")
    print(f"  its consequences withdrawn              : {mean(withdrawn):.3f}   (G5a)")
    print(f"  unrelated consequences that moved       : {mean(collateral):.3f}   (collateral)")
    print(f"  provenance predicted what moved         : {hit / (hit + missed):.3f}   (bookkeeping vs substrate)")
    print(f"  of the {survived} that kept their answer, reached it by a different route: "
          f"{rerouted / max(survived, 1):.3f}   (so not continued influence)")


def main(premises: int = 24, controls: int = 200, path: str = "out/online.pt") -> None:
    universe = generate(0, entities=200, relations=20, out_degree=10)
    corpus, model = build(universe), load(path)
    held = set(online_facts())
    memory = Cached(model, corpus, universe.entities)
    measure(eligible(universe, held), held, memory, zeroing(model, corpus, memory), premises, controls)


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
