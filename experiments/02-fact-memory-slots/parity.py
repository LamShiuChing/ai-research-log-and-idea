"""Is the slot store worth its weights? The G5a suite and four discriminators, on both stores.

`cascade.py` measures belief revision against the slot store. Nothing in the board, the
provenance or the arithmetic ever sees a slot, so the same measurement should hold on any
substrate -- and the cheapest substrate is a dict. Two questions, one run:

- **parity.** Does the upper stack score the same on a dict? If it does, the store is a
  swappable part, and the 226-token vocabulary stops being the project's ceiling.
- **discrimination.** A dict is exact, unbounded, instant and free. The slot store costs a GPU,
  a retrain, and a ceiling of 200 entities. It only deserves to exist if it does something the
  dict cannot, so this asks it on four axes and reports the answer whichever way it falls.

The four axes are the ones PROJECT_GOAL.md actually claims for a learned memory: stay silent on
a question never taught (§17), say how sure it is, score whether a fact is worth writing at all
(§6, glt/admission.py), and answer a fact nobody ever stated (§18's composition claim, asked of
the substrate rather than of the board).

    python parity.py            # dict G5a + the four axes
    python parity.py 8 50       # fewer premises / controls
"""

import sys
from random import Random

import torch

from cascade import Cached, eligible, measure
from chat import DEVICE, MODEL, address, load, slots_of
from glt.admission import surprise_bits
from glt.corpus import build, prefix
from glt.store import Facts
from glt.universe import absences, generate
from perturb import online_facts

#: The deployed, abstaining backbone. `out/online.pt` cannot be used here at all: its
#: vocabulary is 225 words with no token for silence, so half these axes are unaskable of it.
SLOTS = MODEL


def popping(memory):
    """Retract by dropping the key. The thunk puts the object back."""
    def retract(fact):
        old = memory.erase(fact.path[0], fact.subject)
        return lambda: memory.write(fact.path[0], fact.subject, old)
    return retract


@torch.no_grad()
def straight(model, corpus, facts):
    """Ask the substrate itself, with no board in the way: one forward pass, argmax."""
    ids = torch.tensor([corpus.tokenizer.encode(prefix(f)) for f in facts], device=DEVICE)
    best = model(ids)[0][:, -1].float().argmax(-1)
    return [corpus.tokenizer.itos[i] for i in best.tolist()]


def rate(xs) -> float:
    return sum(xs) / len(xs)


def silence(store, facts) -> float:
    """Asked something it was never taught, how often does it decline to name somebody?"""
    return rate([store.follow(f.path[0], f.subject)[0] is None for f in facts])


def doubt(store, known, absent) -> tuple[float, float]:
    """Mean confidence when it is right, and when it answers a question that has no answer."""
    right = [a[1] for f in known if (a := store.follow(f.path[0], f.subject))[0] == f.object]
    wrong = [a[1] for f in absent if (a := store.follow(f.path[0], f.subject))[0] is not None]
    return rate(right or [0.0]), rate(wrong or [0.0])


def unwritten(model, corpus, facts) -> tuple[float, float]:
    """The same facts' MDL bits with their slots written, then emptied, then written back.

    Comparing one fact set against a different one measures the sets. Asking what a fact costs
    when the model holds it against what *that same fact* costs when it does not is the only
    form of the question with nothing else moving.
    """
    written = surprise_bits(model, corpus, facts, DEVICE).tolist()
    saved = {slot: slots_of(model).values[slot].clone()
             for slot in {address(model, corpus, f.path[0], f.subject) for f in facts}}
    with torch.no_grad():
        for slot in saved:
            slots_of(model).values[slot] = 0
        empty = surprise_bits(model, corpus, facts, DEVICE).tolist()
        for slot, row in saved.items():
            slots_of(model).values[slot] = row
    return rate(written), rate(empty)


def main(premises: int = 24, controls: int = 200, n: int = 400) -> None:
    universe = generate(0, entities=200, relations=20, out_degree=10)
    corpus, held = build(universe), set(online_facts())
    facts = Facts({(f.path[0], f.subject): f.object for f in universe.base})

    print(f"=== G5a on a dict ({len(facts.facts)} facts, no vocabulary, no ceiling) ===")
    measure(eligible(universe, held), held, facts, popping(facts), premises, controls)

    print("\n=== four things a dict cannot do, asked of both stores ===")
    model = load(SLOTS)
    slots = Cached(model, corpus, universe.entities)
    rng = Random(0)
    absent = rng.sample(absences(universe), n)
    known = rng.sample(sorted(held, key=lambda f: (f.subject, f.path)), min(n, len(held)))

    print(f"\n1. silence on a question never taught   ({n} absences)")
    print(f"   dict  : {silence(facts, absent):.3f}")
    print(f"   slots : {silence(slots, absent):.3f}")

    print("\n2. a usable doubt signal   (mean confidence when right / when it answers an absence)")
    for name, store in (("dict ", facts), ("slots", slots)):
        right, wrong = doubt(store, known, absent)
        print(f"   {name} : {right:.3f} right, {wrong:.3f} wrong   -> gap {right - wrong:+.3f}")

    print("\n3. is this fact worth writing?   (MDL bits, glt/admission.py)")
    print("   dict  : n/a -- a dict knows membership, not how surprising a new fact is")
    print("   slots : %.2f bits holding the fact, %.2f bits with that same slot emptied"
          % unwritten(model, corpus, known))

    print("\n4. a fact nobody stated   (two-hop, asked of the substrate in one pass, no board)")
    print("   dict  : 0.000 on every split   (by construction -- the key was never written)")
    for split in ("train_inferred", "test_inferred_id", "test_inferred_ood"):
        sample = rng.sample(corpus.heldout[split], min(n, len(corpus.heldout[split])))
        got = rate([a == f.object for a, f in zip(straight(model, corpus, sample), sample)])
        note = "   <- trained on: the memorisation control" if split == "train_inferred" else ""
        print(f"   slots : {got:.3f}  {split}{note}")
    # The decisive row: the splits above compose facts learned in *pretraining*. These compose
    # facts written into slots at inference, which is the only kind a continuous learner gets.
    written = rng.sample([d.fact for d in eligible(universe, held)], n)
    got = rate([a == f.object for a, f in zip(straight(model, corpus, written), written)])
    print(f"   slots : {got:.3f}  premises written into slots at inference   <- the one that matters")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:]))
