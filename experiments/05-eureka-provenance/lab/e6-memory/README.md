# E6 — external memory: kNN output interpolation

The EP3 finding was that trust survives naturalism but *recall* of
unsupervised facts collapses (heldout capitals 4/20). E6 attacks recall
directly: a datastore of (hidden state → next token) pairs built from a
trusted reference corpus (true QA lines, all 113 countries), blended into
generation as `p = λ·p_knn + (1−λ)·p_lm` — kNN-LM style. **No retraining;
pure inference-time surgery on the already-trained EP3 model.**

## Results

2026-08-31, out-epnat (10.65M), datastore 10,397 entries.

| λ | heldout cap /20 | heldout con /20 | train cap /30 | train con /30 |
| --- | --- | --- | --- | --- |
| 0.0 (memory off) | 4 | 10 | 27 | 29 |
| 0.25 | 6 | 10 | 27 | 29 |
| 0.5 | 20 | 19 | 30 | 30 |
| **0.75** | **20** | **20** | **30** | **30** |
| 1.0 | 20 | 20 | 30 | 30 |

Heldout recall goes from collapsed to perfect; train recall *improves* (the
model's 3 own misses get corrected by the store). And knowledge becomes
editable without touching weights:

```
Indonesia: Jakarta -> edited to Asuncion | model now says 'Asuncion'
Singapore: Singapore -> edited to Beijing | model now says 'Beijing'
Guyana: Georgetown -> edited to Vienna   | model now says 'Vienna'
```

Note what λ=1.0 working means: even with the LM's distribution fully
replaced, the *model* still computes the query — the weights' job shifts from
storing facts to understanding questions. Facts live in a curatable store;
the network is the retrieval encoder. This also covers a large slice of
continual learning for free: new facts = append to the store, no retraining,
no catastrophic forgetting.

## Honest limits

The datastore here contains eval-format QA lines — the most favorable
condition. Open stress tests: paraphrased queries vs the store, distractor /
noisy entries, a *poisoned* store (the trust problem relocated to the memory
— PE's anchor lesson applies to the datastore too), and scale.

## Run

```powershell
cd ..\gpt
python ..\e6-memory\knn.py out-epnat
```
