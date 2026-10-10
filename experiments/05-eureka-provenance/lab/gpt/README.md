# lab/gpt — our fork of nanoGPT

Forked from karpathy/nanoGPT @ `3adf61e` (MIT, see LICENSE). This copy is ours
to modify: every component we add to the architecture lands here, tracked and
diffable against upstream. `lab/nanogpt/` stays a pristine gitignored clone
for baselines.

Deviations from upstream so far:

- `train.py`: `--dataset` may be a directory path (e.g.
  `--dataset=../nanogpt/data/facts_char`), so experiments keep one copy of
  their data.

Component queue (one at a time, each vs an unmodified baseline on the same
data):

1. **Confidence head** — second output head predicting p(answer correct),
   trained with a proper scoring rule; explicit self-evaluation channel.
2. **Provenance embeddings** — per-source tags on claims; can the model learn
   source reliability?
3. **External memory** (kNN-style output interpolation) — editable knowledge
   without retraining.
