# EP2 — per-topic trust

Follow-up to [EP](../ep-provenance/README.md), whose sources were globally pure
(always honest or always lying), so one trust scalar per source sufficed. Real
sources are reliable *about some things*: this experiment tests whether the
model can learn trust as a function of **(source, topic)**.

## Design

Two topics per entity (color, size), two tagged sources:

- **S1**: truthful about color, lies about size
- **S2**: lies about color, truthful about size

Votes are balanced 3v3 per entity-topic — no majority signal — and both
sources are exactly 50% reliable globally, so scalar trust is useless. QA
supervises both topics for 400 entities; 100 heldout entities have claims only.

Heldout outcome pattern diagnoses the learned structure:

| color acc | size acc | learned structure |
| --- | --- | --- |
| high | high | per-topic trust |
| high | ~0 (or reverse) | scalar trust (picked a favorite source) |
| ~50 | ~50 | none |

## Run

```powershell
python prepare.py
cd ..\gpt
python train.py ..\e2-abstention\train_facts.py --compile=False --dataset=../nanogpt/data/topic_char --out_dir=out-topic --n_sources=3
python ..\ep2-topic-trust\eval.py out-topic
```

## Results

2026-08-31, 10.64M / 2000 iters.

| | color | size |
| --- | --- | --- |
| train QA | 100/100 | 99/100 |
| **heldout: truth** | **85** | **84** |
| heldout: lie | 5 | 8 |
| heldout: other | 10 | 8 |

The (high, high) pattern: trust localized by topic and generalized to
unsupervised entities on both topics symmetrically — the model did not
collapse to a favorite source (that would read ~high/~0), and balanced votes
rule out majority-counting. The reliability judgment is computed from the
**interaction** of source embedding and sentence topic, somewhere in
attention — a compositional trust structure, learned without any reliability
labels. Noisier than EP's 97% (harder task; ~10% format drift in "other"),
but unambiguous.
