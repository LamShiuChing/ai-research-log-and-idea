# E2 — can training teach a model to know what it doesn't know?

Toy-scale test of the incentive thesis in OpenAI's *Why Language Models
Hallucinate* (arxiv.org/abs/2509.04664): models fabricate because training
rewards guessing over abstaining. We change the training data incentive and
measure whether abstention generalizes.

## Design

900 invented entities, each (potentially) with a color fact.

- **known** (400): facts appear in training
- **distractor** (400): appear in training only as `A: unknown.` (calibrated set only)
- **heldout** (100): never appear in training at all — the probe

Two identical baby GPTs (6 layer / 384 dim, ~10M params), same steps, same data
format, one variable:

| model | trains on | teaches |
| ----- | --------- | ------- |
| baseline | known facts only | always answer |
| calibrated | known facts + distractor "unknown"s | unfamiliar name → say "unknown" |

**Question:** on heldout entities, does the calibrated model *generalize* the
abstention rule to names it has never seen — without forgetting known facts?

- Baseline expected: fabricates a color for every heldout entity (the toy
  version of hallucination).
- Success: calibrated model abstains on most heldout entities while keeping
  known-fact accuracy near baseline. The tradeoff between the two numbers is
  the finding.

## Run

```powershell
python prepare.py
cd ..\nanogpt
python train.py ..\e2-abstention\train_facts.py --compile=False
python train.py ..\e2-abstention\train_facts.py --compile=False --dataset=facts_char_calib --out_dir=out-facts-calib
python ..\e2-abstention\eval.py out-facts-baseline out-facts-calib
```

## Results

2026-08-31, RTX 4080 SUPER, ~3 min total for both trainings.

| model | known correct | known abstained | heldout fabricated | heldout abstained |
| ----- | ------------- | --------------- | ------------------ | ----------------- |
| baseline | 100/100 | 0 | **100/100** | 0 |
| calibrated | 100/100 | 0 | **30/100** | 70/100 |

Identical 10.64M-param models, identical steps and format; the only variable
was 400 training lines answering "unknown" for entities outside the fact set.
Fabrication on never-seen entities dropped 100% → 30% at zero cost to
known-fact recall — the incentive thesis reproduces at toy scale.

Open thread: which 30 heldout names still fool the calibrated model, and why?
