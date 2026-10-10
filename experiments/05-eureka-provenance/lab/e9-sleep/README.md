# E9 — sleep: consolidating the modules into the core

The base can't merge bottleneck adapters arithmetically (nonlinear serial
modules don't fold into weight grids), so consolidation goes through
**verified self-distillation**: each skill module answers thousands of fresh
questions, the oracle keeps only the correct ones, colors replay from the
fact store, and a copy of the base continue-trains on the mixture. Then the
**bare base** — adapters stripped, router gone — takes the four-skill exam.

## Results

2026-08-31. Verification yields during curriculum building: addition
2100/3473 kept (60% — the module's own accuracy), subtraction 2148/2171
(99%), multiplication 2215/2959 (75%). The modules' mistakes never entered
the curriculum.

| skill | module (before) | bare consolidated base |
| --- | --- | --- |
| colors | 100 (base) | **100** |
| addition | 63 | **92** |
| subtraction | 100 | 92 |
| multiplication | 74 (self-taught) | **81** |

Two skills came out of consolidation **better than the modules that taught
them** (63→92, 74→81): the oracle filtered the teachers' noise, so the
full-capacity base learned from a cleaner curriculum than the modules ever
saw — distillation as amplification. Subtraction paid a small consolidation
tax (100→92). Nothing was forgotten. The adapters are now deletable; the
skills live in the weights.

## The life cycle this completes

1. **Fast learning** (daily): facts → store, skills → adapters. Reversible.
2. **Sleep** (periodic): verified self-distillation folds the periphery into
   the core; prune the merged modules. This experiment.
3. **Growth** (rare): new primitives → grow embeddings/layers zero-init.
4. **Rebirth** (generational): full re-pretraining; the store and logs are
   the curriculum.

## Run

```powershell
cd ..\gpt
python ..\e9-sleep\prepare_sleep.py out-e7-a out-e7-adapter out-e7-adapter-c out-e8-adapter
# copy out-e7-a/ckpt.pt to out-e9-base/, then:
python train.py ..\e2-abstention\train_facts.py --compile=False --dataset=../nanogpt/data/e9_sleep --out_dir=out-e9-base --init_from=resume --max_iters=4500 --lr_decay_iters=4500 --always_save_checkpoint=True
python ..\e9-sleep\eval.py out-e9-base
```
