# lab — model-internals track

Architecture experiments on nanoGPT (github.com/karpathy/nanoGPT): the model is
a few hundred lines of PyTorch in `nanogpt/model.py`, built to be modified.

Models trainable on one 16GB GPU won't beat modern LLMs, but they let you
*compare ideas* cheaply, with full control of data, objective and
architecture — which is the whole research program of this repo.

## Method — one experiment at a time

1. Baseline first: train unmodified nanoGPT on Shakespeare (char-level), record
   final val loss. Minutes on a 4080.
2. Change exactly one thing in `model.py`. First candidates:
   - positional encoding: learned embeddings → RoPE
   - normalization: LayerNorm → RMSNorm, pre-norm placement
   - attention: head count, or a sliding-window variant
3. Retrain with identical config/seed, compare val loss curves.
4. Keep a one-line result log here. An idea only counts if it beats baseline
   at equal parameter count and steps.

## Setup

```powershell
cd lab/nanogpt   # cloned copy, gitignored; upstream is the source of truth
pip install torch numpy transformers datasets tiktoken wandb tqdm
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
```

(Install list is from the nanoGPT README — check it after cloning in case
upstream changed.)

## Results log

| change | val loss | vs baseline |
| ------ | -------- | ----------- |
| (baseline) | — | — |
