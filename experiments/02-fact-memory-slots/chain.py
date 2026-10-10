"""Symbolic chaining over the slot memory -- the P4 derivation loop in its smallest form.

PROJECT_GOAL.md §17 measured multi-hop composition over slot-stored facts at 0.009, flat, and
identified the cause as structural: answering `r2 of r1 of X` needs the address of the *second*
lookup, which is a function of the *first* lookup's answer, and a single forward pass has
nowhere to put that intermediate.

A chainer has somewhere to put it. It asks the model one hop at a time and carries the answer
forward:

    r1 of X  ->  Y        (one model query)
    r2 of Y  ->  Z        (a second query, addressable only because Y is now known)

Everything it knows comes from the model: each hop is a normal forward pass through the same
memory, and no ground truth is consulted. The only thing added is the decomposition.
"""

import sys

import torch

from glt.corpus import build, prefix
from glt.model import GPTConfig
from glt.questions import cooccurring
from glt.universe import generate
from perturb import online_facts

sys.path.insert(0, ".")
from chat import DEVICE, answer, load  # noqa: E402  -- reuses the demo's single-hop query

torch.serialization.add_safe_globals([GPTConfig])


def chain(model, corpus, fact):
    """Walk the relation path one hop at a time, carrying each answer into the next address."""
    current, trace = fact.subject, []
    for relation in fact.path:
        current, confidence = answer(model, corpus, relation, current)
        trace.append((relation, current, confidence))
    return current, trace


@torch.no_grad()
def direct(model, corpus, fact):
    """What the transformer answers in a single pass -- the 0.009 baseline."""
    ids = torch.tensor([corpus.tokenizer.encode(prefix(fact))], device=DEVICE)
    return corpus.tokenizer.itos[model(ids)[0][0, -1].argmax().item()]


def main(path="out/online.pt"):
    universe = generate(0, entities=200, relations=20, out_degree=10)
    corpus, model = build(universe), load(path)
    held = set(online_facts())
    adjacent = cooccurring(universe)

    # Exactly the `online_composition` set of PROJECT_GOAL.md §17: consequences of the facts
    # written into slots at inference, never shown in any form during training.
    questions = [d.fact for d in universe.derived
                 if not d.premises.isdisjoint(held) and d.fact.object not in adjacent[d.fact.subject]]

    one_pass = sum(direct(model, corpus, f) == f.object for f in questions)
    chained = sum(chain(model, corpus, f)[0] == f.object for f in questions)
    hops = sum(answer(model, corpus, f.path[0], f.subject)[0] ==
               next(iter([g.object for g in universe.base
                          if g.subject == f.subject and g.path == (f.path[0],)]), None)
               for f in questions[:500])

    print(f"questions (2-hop, consequences of slot-written facts): {len(questions)}")
    print(f"  transformer, single pass : {one_pass/len(questions):.3f}")
    print(f"  symbolic chain, two hops : {chained/len(questions):.3f}")
    print(f"  (first hop alone, sanity): {hops/min(500, len(questions)):.3f}")

    print("\nworked example:")
    example = questions[0]
    final, trace = chain(model, corpus, example)
    print(f"  asked   : {example.phrase()} is ?   (truth: {example.object})")
    for relation, result, confidence in trace:
        print(f"  hop     : {relation} of ... -> {result}  ({confidence:.0%})")
    print(f"  chained : {final}   {'CORRECT' if final == example.object else 'wrong'}")
    print(f"  one pass: {direct(model, corpus, example)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
