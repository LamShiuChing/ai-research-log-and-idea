# E2b — can a model flag its own contested knowledge?

Follow-up to [E2](../e2-abstention/README.md). E2 showed "unfamiliar → unknown"
is trainable. But calibration can't help with *learned falsehoods* — a
well-learned lie feels exactly like a well-learned truth. This experiment asks
the nearest trainable question: when the training evidence itself **conflicts**,
can the model learn to say "disputed" — and does that policy generalize to
entities whose QA behavior it was never taught?

## Design

Training text has two layers, mimicking pretraining vs post-training:

- **claims** (the corpus): `The color of <name> is <color>.` — 10 agreeing
  copies for consistent entities; 6 vs 4 conflicting copies for contested ones.
- **QA policy**: `Q: ...? A: <color>.` for 300 consistent entities;
  `A: disputed.` for 300 contested entities.

100 consistent + 100 contested **heldout** entities appear in claims only —
no QA lines. At eval, the model must derive the answering policy for them from
its own stored evidence.

Readout on contested heldout:
- **disputed** — detected its own conflicting evidence (success)
- **majority color** — swallowed the fake news (the failure this probes)
- minority / other

## Run

```powershell
python prepare.py
cd ..\nanogpt
python train.py ..\e2-abstention\train_facts.py --compile=False --dataset=contested_char --out_dir=out-contested
python ..\e2b-contested\eval.py out-contested
```

## Results

2026-08-31, same 10.64M model/config as E2, ~2.5 min.

| | consistent (n=100) | contested (n=100) |
| --- | --- | --- |
| trained QA | 100 correct | 100 disputed |
| **heldout (no QA ever)** | 68 correct, **32 falsely disputed** | **58 disputed**, 31 majority color, 9 minority, 2 other |

The policy itself is learned perfectly. Generalization is real but noisy:
58% hits vs a 32% false-alarm floor. Note the confound the floor exposes —
heldout entities of *both* classes lack QA history, and that novelty alone
pushes the model toward "disputed"; true conflict detection is the +26 points
above the floor. Compare E2's clean 70/0: familiarity ("have I seen this
name?") is an easy introspection; evidence variance ("did my sources agree?")
is a much harder one. Data incentives alone get a weak signal through — the
open question is whether an explicit architectural channel sharpens it.
