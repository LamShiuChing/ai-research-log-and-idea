# Sleep on its own dreams: a continual GPT

Can a GPT learn new kinds of English day after day without forgetting the old ones?

- `cgpt.py`: nanoGPT redesigned with a slot "cortex" and **sleep**. Before each night the model dreams 512 texts
  from its own copy; during the night it also keeps its old predictions on those dreams (KL to the copy). No old
  text is stored. Predictions are written in the docstring before the first run.
- `cgpt_gap.py`: stronger sleep (higher KL weight and more) to close the gap to the all-at-once ceiling. Together
  with `cgpt.py`, this is the run behind the 0.58 → 0.08 result (best of 4 pre-defined arms, 3 seeds).
- `twenty_days.py`: 20-day streams on CIFAR-100 (5 new classes a day, frozen ResNet-18) and 20 Newsgroups
  (1 topic a day). A running-sum "link" memory vs a normal network vs all-at-once training. Results are in the
  docstring.
- `FINAL_REPORT.md`: the full project report, including what failed.

**Headline results** (from `FINAL_REPORT.md` and `twenty_days.py`):

| Test | Normal | With the idea |
|---|---|---|
| GPT forgetting, bits/byte (lower is better) | 0.58 | 0.19 with sleep on its own dreams; 0.08 with stronger sleep |
| CIFAR-100: day-1 classes after 20 days | 0.00 | 0.79 with running-sum links |
| 20 Newsgroups: day-1 topics after 20 days | 0.00 | 0.62 with running-sum links |

```bash
pip install torch tiktoken          # cgpt.py
python cgpt.py                      # needs BabyLM 10M "clean" text, see docstring
pip install torch torchvision scikit-learn datasets
python twenty_days.py               # downloads CIFAR-100 / 20 Newsgroups
```
