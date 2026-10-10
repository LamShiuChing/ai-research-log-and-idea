"""Does the language specialist need to be *big*?

Scaling laws are claims about language-modelling loss over natural text: loss falls as a power
law in parameters, data and compute, and "larger models are significantly more sample-efficient"
([Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)); for a fixed compute budget parameters
and tokens should scale together, "for every doubling of model size the number of training
tokens should also be doubled" ([Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)).

This parser has a much narrower job than language modelling -- find the content words of a noun
phrase and put them in canonical order -- and extract.py already showed that *wording diversity*
moves its held-out score a long way at fixed size (0.445 -> 0.895 from 4 to 32 wordings). So the
question is which of the three axes actually binds here, and it is cheap to ask directly.

Held fixed: 8 training wordings, the regime where extract.py measured 0.505 / 0.502 on unseen
wordings, so there is room to move in either direction. Swept: parameters at fixed steps, then
steps at fixed parameters -- the second because this project has already seen a model fit its
training data at 2,500 steps and only generalise near 100,000 (PROJECT_GOAL.md §15).

    python capacity.py            # ~35 min on a 4080
"""

import sys

from extract import DEVICE, HELD_FRAMES, cell
from glt.language import FRAMES, split, train, vocabulary
from glt.universe import generate

#: (n_layer, n_head, n_embd) -- two orders of magnitude of capacity at identical data.
SIZES = ((2, 2, 64), (4, 4, 128), (6, 6, 192), (8, 8, 384))
STEPS = (4_000, 20_000, 100_000)
WORDINGS = 8


def main(n: int = 400) -> None:
    universe = generate(0, entities=200, relations=20, out_degree=10)
    tokenizer = vocabulary(universe.entities, universe.relations)
    train_frames, held_frames = split(FRAMES, HELD_FRAMES)
    fit = dict(tokenizer=tokenizer, frames=train_frames[:WORDINGS], entities=universe.entities,
               relations=universe.relations, head="point", device=DEVICE, log=lambda *_: None)
    scores = lambda model, frames: [cell(model, tokenizer, frames, universe.entities,
                                         universe.relations, depth, n) for depth in (1, 2)]

    def row(label, model):
        trained, held = scores(model, train_frames[:WORDINGS]), scores(model, held_frames)
        print(f"{label:>12}  {sum(p.numel() for p in model.parameters()) / 1e6:8.2f}  "
              + "  ".join(f"{s:7.3f}" for s in trained + held), flush=True)

    print(f"held-out wording accuracy, {WORDINGS} wordings trained on   (n={n} per cell)")
    print(f"{'':>12}  {'params M':>8}  {'fit 1hop':>7}  {'fit 2hop':>7}  {'new 1hop':>7}  {'new 2hop':>7}")

    print("-- capacity, at 4,000 steps --", flush=True)
    for n_layer, n_head, n_embd in SIZES:
        row(f"{n_layer}L x {n_embd}", train(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                                            steps=4_000, **fit))

    print("-- steps, at 4 layers x 128 --", flush=True)
    for steps in STEPS:
        row(f"{steps:,}", train(steps=steps, **fit))


if __name__ == "__main__":
    main(**{k: int(v) for k, v in (a.split("=", 1) for a in sys.argv[1:])})
