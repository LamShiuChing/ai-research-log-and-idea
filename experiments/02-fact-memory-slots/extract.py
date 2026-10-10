"""Step 1 of HANDOFF.md §4: does the language specialist turn free phrasing into a Query?

The gate is paraphrase invariance -- every wording of one question must produce the *same*
Query, because that Query determines the slot address (glt/language.py). Anything less and the
question reaches a different slot, which is indistinguishable from the memory not holding it.

Four measurements, in the order the failures were found:

1. **heads.** Generation against pointing, on words held out of the parser's training data.
   Generation cannot emit a token whose output embedding never took a gradient; pointing names
   a *position* and copies whatever sits there. This is the session-2 addressing lesson again.
2. **wordings.** Held-out wordings were the binding constraint for both heads once the copy
   problem was fixed, so the training-wording count is swept against a fixed held-out set. A
   curve that climbs says the limit is corpus diversity -- which is what a pretrained language
   model buys and a from-scratch 0.9M parser does not.
3. **shapes.** Each operation of the query language, plus `none`: the reject class. Without a
   reject the pointer always points somewhere, so "how many people report to Ben" comes back as
   a confident answer to a question nobody asked.
4. **invariance**, the gate: every phrasing of one query agrees, and the agreed Query is right.

Content words are held out only as a diagnostic. The deployed vocabulary is closed -- a fact
can only be about a token the backbone already has -- so the saved parser trains on all of
them, and wording is the only axis a user can actually vary at runtime.

    python extract.py                 # ~7 min on a 4080
    python extract.py steps=8000 n=800
"""

import sys
from itertools import product
from random import Random

import torch

from glt.language import (FRAMES, NOUN_PHRASES, OPS, SHAPES, arity, extract, phrasings, render,
                          render_shape, save, split, train, vocabulary)
from glt.universe import generate
from glt.workspace import count, follow, reverse, same

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "out/parser.pt"
HELD_FRAMES = 8


def queries(rng: Random, entities, relations, depth: int, n: int):
    """`n` (subject, distractor, path) triples; the distractor is a name that must be ignored."""
    return [(*rng.sample(entities, 2), tuple(rng.choices(relations, k=depth))) for _ in range(n)]


def cell(model, tokenizer, frames, entities, relations, depth: int, n: int, seed: int = 1) -> float:
    """Exact match of the whole Query, one random `follow` phrasing each."""
    rng = Random(seed)
    asked = [render(rng.choice(frames), rng.choice(NOUN_PHRASES), subject, path, distractor)
             for subject, distractor, path in queries(rng, entities, relations, depth, n)]
    got = extract(model, tokenizer, [question for question, _, _ in asked], DEVICE)
    return sum(out == follow(*to_pair(canonical)) for out, (_, canonical, _) in zip(got, asked)) / n


def to_pair(canonical: str):
    words = canonical.split()
    return words[-2], tuple(reversed(words[:-2:2]))


def expected(op: str, words):
    """The Query a wording of `op` over `words` must produce; None for the reject class."""
    if op == "reverse":
        return reverse(words[0], words[1])
    if op == "count":
        return count(reverse(words[0], words[1]))
    if op == "same":
        return same(follow(words[0], (words[1],)), follow(words[2], (words[3],)))
    return None


def shape_cell(model, tokenizer, op: str, entities, relations, n: int, seed: int = 3) -> float:
    """Exact-Query accuracy for one non-`follow` shape. For `none`, that means answering None."""
    rng = Random(seed)
    asked = []
    for _ in range(n):
        frame = rng.choice(SHAPES[op])
        words = [rng.choice(entities if i % 2 == 0 else relations) for i in range(arity(frame))]
        asked.append((render_shape(frame, words, rng.choice(entities))[0], expected(op, words)))
    got = extract(model, tokenizer, [question for question, _ in asked], DEVICE)
    return sum(out == want for out, (_, want) in zip(got, asked)) / n


def invariance(model, tokenizer, frames, entities, relations, depth: int, n: int, seed: int = 2):
    """Fraction of queries whose every phrasing agrees, and whose agreed Query is also right."""
    rng = Random(seed)
    items = queries(rng, entities, relations, depth, n)
    asked = [phrasings(subject, path, distractor, frames) for subject, distractor, path in items]
    got = extract(model, tokenizer, [q for group in asked for q in group], DEVICE)
    width = len(frames) * len(NOUN_PHRASES)
    groups = [got[i * width : (i + 1) * width] for i in range(n)]
    truth = [follow(subject, path) for subject, _, path in items]
    return (sum(len(set(group)) == 1 for group in groups) / n,
            sum(set(group) == {answer} for group, answer in zip(groups, truth)) / n)


def report(model, tokenizer, splits, words, n: int) -> None:
    print(f"{'wordings':>9}  {'words':>9}  {'1 hop':>7}  {'2 hop':>7}")
    for (fname, frames), (wname, entities, relations) in product(splits, words):
        scores = (cell(model, tokenizer, frames, entities, relations, depth, n) for depth in (1, 2))
        print(f"{fname:>9}  {wname:>9}  " + "  ".join(f"{s:7.3f}" for s in scores))


def main(steps: int = 4000, n: int = 400) -> None:
    universe = generate(0, entities=200, relations=20, out_degree=10)
    tokenizer = vocabulary(universe.entities, universe.relations)
    train_frames, held_frames = split(FRAMES, HELD_FRAMES)
    train_entities, held_entities = split(universe.entities, 40)
    train_relations, held_relations = split(universe.relations, 4)
    splits = (("trained", train_frames), ("held out", held_frames))
    words = (("trained", train_entities, train_relations), ("held out", held_entities, held_relations))
    fit = dict(tokenizer=tokenizer, steps=steps, device=DEVICE)

    print(f"{len(FRAMES)} follow wordings, {HELD_FRAMES} held out:")
    for frame in held_frames:
        print(f"  {frame}")

    print("\n=== 1. output head, at full wording diversity ===")
    for head in ("generate", "point"):
        model = train(frames=train_frames, entities=train_entities, relations=train_relations,
                      head=head, **fit)
        report(model, tokenizer, splits, words, n)

    print("\n=== 2. how many wordings does a held-out wording need? (pointer head) ===")
    print(f"{'trained on':>10}  {'1 hop':>7}  {'2 hop':>7}  {'agree':>7}  {'correct':>7}")
    for wordings in (4, 8, 16, 32):
        model = train(frames=train_frames[:wordings], entities=train_entities,
                      relations=train_relations, head="point", **fit)
        scores = [cell(model, tokenizer, held_frames, train_entities, train_relations, depth, n)
                  for depth in (1, 2)]
        agree, correct = invariance(model, tokenizer, held_frames, train_entities,
                                    train_relations, 2, n=100)
        print(f"{wordings:>10}  " + "  ".join(f"{s:7.3f}" for s in (*scores, agree, correct)))

    print("\n=== 3. the parser to deploy: every wording, every word ===")
    model = train(frames=FRAMES, entities=universe.entities, relations=universe.relations,
                  head="point", **fit)
    print("  by shape of question:")
    for op in OPS:
        score = (cell(model, tokenizer, FRAMES, universe.entities, universe.relations, 2, n)
                 if op == "follow" else
                 shape_cell(model, tokenizer, op, universe.entities, universe.relations, n))
        print(f"    {op:>8}  {score:.3f}" + ("   (an unanswerable question, correctly refused)"
                                             if op == "none" else ""))
    print(f"  paraphrase invariance over all {len(FRAMES) * len(NOUN_PHRASES)} follow phrasings:")
    for depth in (1, 2):
        agree, correct = invariance(model, tokenizer, FRAMES, universe.entities,
                                    universe.relations, depth, n=200)
        print(f"    {depth} hop: agree {agree:.3f}   agree and correct {correct:.3f}")

    save(model, tokenizer, OUT)
    print(f"  saved {OUT}")


if __name__ == "__main__":
    main(**{k: int(v) for k, v in (a.split("=", 1) for a in sys.argv[1:])})
