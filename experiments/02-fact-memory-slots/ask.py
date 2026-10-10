"""Step 3 of HANDOFF.md §4: the whole system, from free-form English to an answer.

Same questions as chain.py -- the 2-hop consequences of facts written into slots at inference,
never seen in any form during training -- but asked in English rather than handed to the system
already decomposed. Every part is now doing its own job:

    language   an arbitrary phrasing -> the canonical token line   (glt/language.py)
    logic      that line -> one hop at a time, with provenance     (glt/workspace.py)
    memory     one hop -> one forward pass through the slots       (chat.answer)

chain.py measured 0.010 for a single forward pass and 1.000 for the same model asked hop by
hop. The number that matters here is whether that 1.000 survives having to read the question,
and where the losses land when it does not.

    python ask.py
"""

import sys
from random import Random

import torch

from glt.corpus import build
from glt.language import FRAMES, NOUN_PHRASES, extract, render
from glt.model import GPTConfig
from glt.questions import cooccurring
from glt.universe import generate
from glt.workspace import follow, solve, trace
from perturb import online_facts

sys.path.insert(0, ".")
from chat import DEVICE, Slots, load  # noqa: E402  -- the memory specialist
from glt.language import load as load_parser  # noqa: E402

torch.serialization.add_safe_globals([GPTConfig])


def phrase(rng, universe, fact):
    """One arbitrary English wording of a question, with an unrelated name planted in it."""
    return render(rng.choice(FRAMES), rng.choice(NOUN_PHRASES), fact.subject, fact.path,
                  rng.choice(universe.entities))[0]


def main(path: str = "out/online.pt", parser: str = "out/parser.pt") -> None:
    universe = generate(0, entities=200, relations=20, out_degree=10)
    corpus, model = build(universe), load(path)
    reader, tokenizer = load_parser(parser, DEVICE)
    adjacent, held = cooccurring(universe), set(online_facts())
    questions = [d.fact for d in universe.derived
                 if not d.premises.isdisjoint(held) and d.fact.object not in adjacent[d.fact.subject]]

    rng = Random(0)
    asked = [phrase(rng, universe, fact) for fact in questions]
    parsed = extract(reader, tokenizer, asked, DEVICE)
    memory = Slots(model, corpus, universe.entities)

    read = sum(query == follow(fact.subject, fact.path) for query, fact in zip(parsed, questions))
    boards = [solve(query, memory) if query else ([], None) for query in parsed]
    correct = sum(note is not None and note.answer == fact.object
                  for (_, note), fact in zip(boards, questions))

    print(f"questions (2-hop, consequences of slot-written facts): {len(questions)}")
    print(f"  read correctly (canonical form recovered) : {read / len(questions):.3f}")
    print(f"  answered correctly, end to end            : {correct / len(questions):.3f}")
    print(f"  answered correctly given a correct read   : "
          f"{sum(note is not None and note.answer == f.object for (_, note), f, q in zip(boards, questions, parsed) if q == follow(f.subject, f.path)) / max(read, 1):.3f}")

    board, note = boards[0]
    print(f"\nworked example:\n  asked   : {asked[0]}")
    print(f"  read as : {parsed[0]}")
    for step in trace(board, note):
        print(f"  hop     : {step}  ({step.confidence:.0%})")
    print(f"  answer  : {note.answer}   (truth {questions[0].object})")


if __name__ == "__main__":
    main(*sys.argv[1:])
