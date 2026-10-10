# EP3-natural — provenance trust meets real data

Same source structure as [EP2](../ep2-topic-trust/README.md) (S1 truthful on
capitals / lies on continents, S2 reversed; balanced claims), with the
naturalism knobs turned: 113 real countries, real capitals/continents,
plausible lies (real capitals of other countries), four claim templates and
two QA templates per topic in natural English. 93 countries QA-supervised,
20 heldout. A local chat app (`app/serve.py`) exposes the trained model for
human-feedback probing.

## Results

2026-08-31, 10.65M / 2000 iters.

| | capital | continent |
| --- | --- | --- |
| train QA (n=50) | 47 | 49 |
| heldout: truth (n=20) | 4 | 10 |
| heldout: lie | **0** | **3** |
| heldout: other | 16 | 7 |

Two separate readings, and they diverge:

- **Trust survives naturalism.** Lie rates on heldout stay near floor (0/20,
  3/20) — the model is *not* eating the misinformation, on either topic.
- **Unsupervised recall does not.** "Other" dominates heldout capitals: with
  ~90 distinct multi-word capital strings at char level (vs 10 colors in EP2),
  generating the exact unsupervised fact is the bottleneck, not gullibility.
  Continents (6 values) sit in between: 50% truth, 15% lie.

The scaling lesson: the provenance mechanism transfers; open-vocabulary
generation at 10M-char-level is what breaks first. Next rungs: word-level
tokens, wider model, more claim repetitions — the shape of a rented-GPU run.

## Run

```powershell
python prepare.py
cd ..\gpt
python train.py ..\e2-abstention\train_facts.py --compile=False --dataset=../nanogpt/data/natural_char --out_dir=out-epnat --n_sources=3
python ..\ep3-natural\eval.py out-epnat
python ..\..\app\serve.py    # chat probe on http://localhost:8341
```
