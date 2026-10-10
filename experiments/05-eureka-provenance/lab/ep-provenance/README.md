# EP — provenance embeddings: learning whom to trust

Component 2 in the [fork](../gpt/README.md). Standard pretraining discards
where text came from, so a model facing coordinated lies can only count votes.
This experiment keeps provenance: every token carries a source id, embedded and
added to the input sum (`wse` in `model.py`, enabled by `--n_sources`).

## Design

500 entities. Per entity, five claims: 2 truthful sources say the true color,
3 liar sources agree on the same wrong color — **the majority is always the
lie** (9 lines vs 6 after reps). QA (source 0) teaches ground truth for 400
entities. 100 heldout entities appear in claims only.

- **tagged model** (`--n_sources=7`): can fit QA truth by learning that
  sources 1–3 are reliable — and, if it generalizes, answers heldout entities
  with the *minority-but-reliable* color.
- **control** (`--n_sources=0`): same tokens, no provenance — can only follow
  the majority, i.e. eat the fake news.

The component adds no judgment; it restores **access to a signal** (source
identity) that the data had and standard training throws away.

## Run

```powershell
python prepare.py
cd ..\gpt
python train.py ..\e2-abstention\train_facts.py --compile=False --dataset=../nanogpt/data/prov_char --out_dir=out-prov --n_sources=7
python train.py ..\e2-abstention\train_facts.py --compile=False --dataset=../nanogpt/data/prov_char --out_dir=out-prov-ctrl
python ..\ep-provenance\eval.py out-prov out-prov-ctrl
```

## Results

2026-08-31, both models 10.64M / 2000 iters on identical tokens.

| model | train QA | heldout: truth (minority, reliable) | lie (majority) | other |
| --- | --- | --- | --- | --- |
| **tagged** (`--n_sources=7`) | 100/100 | **97** | 1 | 2 |
| control (no tags) | 94/100 | 29 | **56** | 15 |

No one ever labeled the sources. The tagged model inferred which sources are
reliable purely from the coherence between claims and QA ground truth on the
supervised entities — then **generalized that trust to 97/100 entities it had
no QA supervision for**, siding with the outvoted-but-reliable minority.
The control did exactly what vote-counting predicts: ate the coordinated lie
(56%) and couldn't even fully fit the supervised QA (94/100) because
conflicting anonymous claims left it no way to reconcile.

The component added no judgment anywhere — it restored access to metadata
(origin) that standard training discards, and reliability was *learned*.

Caveats: pure-liar/pure-truthful sources, one fact type, fixed 9v6 vote ratio,
toy scale. Real sources are mixed-reliability per topic; per-topic trust is
the natural follow-up.
