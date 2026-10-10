# EP4a — anchor poisoning dose-response

PE learns trust *relative to the QA anchor*. This experiment corrupts a
fraction X of the 400 anchor entities (their QA answer becomes the coordinated
lie) and measures heldout truth-vs-lie as the anchor rots. Rig = EP (2
truthful sources vs 3 coordinated liars); claims identical across X, only the
anchor changes.

## Results

2026-08-31, four 10.64M models, identical configs.

| anchor poisoned | heldout truth | heldout lie | other |
| --- | --- | --- | --- |
| 0% | **100** | 0 | 0 |
| 10% | 92 | 4 | 4 |
| 25% | 69 | 20 | 11 |
| 50% | 46 | **47** | 7 |

The curve is graceful and roughly proportional — no cliff, no sudden
allegiance flip. At 50% the two coalitions are equally anchor-consistent and
the model lands on a statistical coin flip, exactly as theory predicts.

Two readings of the same line:

- **Robustness**: 10% anchor rot costs only 8 points — PE does not need a
  perfect anchor, it needs *majority* anchor integrity.
- **Vulnerability**: there is also no safe threshold — every percent of anchor
  corruption buys the adversary a roughly proportional slice of trust.
  Trust ≈ anchor coherence, linearly, in both directions.

## Run

```powershell
python prepare.py
cd ..\gpt
# for X in 0 10 25 50:
python train.py ..\e2-abstention\train_facts.py --compile=False --dataset=../nanogpt/data/prov4a_xX --out_dir=out-4a-xX --n_sources=7
python ..\ep4a-anchor\eval.py out-4a-x0 out-4a-x10 out-4a-x25 out-4a-x50
```
