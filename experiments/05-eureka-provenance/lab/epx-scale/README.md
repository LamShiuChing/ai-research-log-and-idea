# EPX — largest local run: trust-weighted recall emerges with scale

EP3's structure scaled to the 4080's comfortable maximum: 89.46M params
(GPT-2-small dimensions), word-level tokens (vocab 5,867), 5,012 entities
(5,000 synthetic + 12 real), provenance channel on. 4,010 entities
QA-anchored, 1,002 heldout. S1 truthful capitals / lies continents, S2
reversed; balanced claims.

## Results — the emergence trajectory

Three checkpoints of the same run (heldout n=200 sampled):

| iters | train loss | train cap/con | heldout capital T/L/other | heldout continent T/L/other |
| --- | --- | --- | --- | --- |
| 3,000 | 1.41 | 94/21 | 0 / 0 / 200 | 30 / 39 / 131 |
| 12,000 | 0.94 | 100/28 | 159 / 0 / 41 | 35 / 29 / 136 |
| 25,000 | 0.25 | **100/100** | **191 / 0 / 9** | **199 / 1 / 0** |

At full convergence the model recalls facts it had **zero QA supervision
for** — reconstructed purely from balanced, conflicting, provenance-tagged
claims — at 95.5% (capitals) and 99.5% (continents), with essentially zero
lie adoption. Compare the identical experiment at 10M/char (EP3): 4/20.
**Provenance trust doesn't just survive scale; it strengthens with it.**

Notes: the interim continent weakness was pure undertraining (shared 6-way
labels converge later than unique entity-capital bindings). Val loss (2.98)
diverges from train loss (0.25) — expected in the memorization regime; the
generalization that matters (heldout entities) is measured directly above.
Training time: ~15 min total on the RTX 4080 SUPER.
