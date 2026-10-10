# E5 — continual learning: forgetting, replay, and the store

Can the model learn 200 new facts (task B) after mastering 200 old ones
(task A)? Four arms; QA-only corpora; same 10.64M config throughout.

## Results

2026-08-31.

| arm | phase 2 | task A /100 | task B /100 |
| --- | --- | --- | --- |
| base | — | 100 | 5 |
| naive continual | train on B only (1000 iters) | **15** | 100 |
| replay | train on B + 25% A replay | 100 | 100 |
| **kNN store** | **none — B appended to the E6 datastore** | **100** | **100** |

- **Catastrophic forgetting, demonstrated**: naive phase-2 training destroyed
  85 points of task A while learning B. This is why deployed LLMs can't just
  "keep training".
- **Replay works** at this scale — at the cost of retraining plus curating old
  data forever.
- **The store ties replay with zero gradient steps**: new facts appended to
  the E6 datastore, weights untouched, nothing forgotten, learning instant.

Design conclusion for the project: **weights learn skills (slow, by
training); the store learns knowledge (instant, by appending).** For factual
continual learning, the memory component from E6 *is* the mechanism; gradient
training is reserved for capabilities the store can't hold (formats,
reasoning patterns, language itself).

Caveat: facts-only tasks. Continual *skill* learning (new formats, new
reasoning) still requires training-based approaches and remains open.

## Run

```powershell
python prepare.py
cd ..\gpt
python train.py ..\e2-abstention\train_facts.py --compile=False --dataset=../nanogpt/data/e5_a --out_dir=out-e5-a
# copy out-e5-a to out-e5-naive and out-e5-replay, then:
python train.py ..\e2-abstention\train_facts.py --compile=False --dataset=../nanogpt/data/e5_b --out_dir=out-e5-naive --init_from=resume --max_iters=3000 --lr_decay_iters=3000 --always_save_checkpoint=True
python train.py ..\e2-abstention\train_facts.py --compile=False --dataset=../nanogpt/data/e5_replay --out_dir=out-e5-replay --init_from=resume --max_iters=3000 --lr_decay_iters=3000 --always_save_checkpoint=True
python ..\e5-continual\eval.py out-e5-a out-e5-naive out-e5-replay
```
