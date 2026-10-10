# EP4b — coverage betrayal (v2 verdict: coverage holes fail OPEN, 80%+)

Does trust earned on an anchored topic bleed into an unanchored one? Rig =
EP2 (S1 truthful color / lies size, S2 reversed, balanced claims) with one
change: **the QA anchor covers only color — no size QA exists anywhere.**

## Results

2026-08-31, 10.64M / 2000 iters.

| probe | truth | lie | other |
| --- | --- | --- | --- |
| color heldout (sanity) | 91 | 2 | 7 |
| size, train entities | 0 | 0 | **100** |
| size, heldout entities | 0 | 0 | **100** |

Sanity holds (EP2-level color trust). But the size rows aren't measuring
trust at all — raw inspection shows the model answers size questions **with
colors** ("Q: What is the size of X? A: blue."). With zero size QA in
training, the model learned "the answer slot contains a color"; the
unanchored topic never reaches the trust machinery because the *ability to
answer it* was never taught.

So the honest finding is narrower and different from the plan: at this scale,
**a coverage hole fails closed** — the model produces off-topic babble, not
confident source-borrowed lies. The anchor teaches two things at once: whom
to trust *and* how to answer. Caution against over-reading: a large
pretrained model has generic QA ability, so at scale a coverage hole likely
*would* be expressible — and could then fail open. The betrayal question
needs EP4b-v2: anchor the size QA *format* with a few neutral entities
(claims balanced between sources) without anchoring size *reliability*, then
re-ask whether S1's color-earned trust leaks into its size lies.

## v2 — format anchored, reliability not

`prepare_v2.py` adds 20 *format entities*: both sources state their size
truthfully and they get size QA — the size-QA format becomes expressible while
carrying zero size-reliability signal (QA agrees with both sources equally
there). Regular 500 entities unchanged: conflicting size claims, no size QA.

### v2 results (2026-08-31, out-4b2)

| probe | truth (S2) | lie (S1) | other |
| --- | --- | --- | --- |
| format entities size QA (sanity) | 20/20 correct | | |
| color heldout (sanity) | 95 | 1 | 4 |
| **size, regular train entities** | 5 | **80** | 15 |
| **size, heldout entities** | 5 | **82** | 13 |

With the format expressible, the coverage hole flips from fail-closed to
**fail-open**: the model fills unanchored answers with the lies of the source
whose *color* record earned its trust. No size-reliability asymmetry exists
anywhere in the data (the format entities validate both sources equally), so
the 80%+ preference for S1's size claims can only be color-earned trust
extrapolating.

Read with EP2, the full picture: anchor coverage of a topic is what *forces*
trust to localize; absent coverage, the model adopts the simplest hypothesis
consistent with the anchor — global trust — and the trusted source gets
believed precisely where nobody audited it. That is the credibility-laundering
attack (build a record in checkable domains, lie in uncheckable ones),
reproduced in weights.

## Run

```powershell
python prepare.py          # v1 (format collapse)
python prepare_v2.py       # v2
cd ..\gpt
python train.py ..\e2-abstention\train_facts.py --compile=False --dataset=../nanogpt/data/cover_char --out_dir=out-4b --n_sources=3
python train.py ..\e2-abstention\train_facts.py --compile=False --dataset=../nanogpt/data/cover2_char --out_dir=out-4b2 --n_sources=3
python ..\ep4b-coverage\eval.py out-4b
python ..\ep4b-coverage\eval_v2.py out-4b2
```
