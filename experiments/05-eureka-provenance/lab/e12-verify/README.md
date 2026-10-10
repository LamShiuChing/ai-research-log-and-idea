# E12 — the verification hierarchy, measured

Three experiments testing the grades that license assertion and learning:
G0 execution oracles, G1 anchored source trust, G2 self-consistency,
G3u user-taught, G4 quarantine.

## E12a — self-consistency as a store-write gate (G2)

The EP3 model answers each question 8× at temperature 0.8; majority answer =
candidate write, agreement = gate.

| set | τ=8/8 accepts | correct among accepted |
| --- | --- | --- |
| memorized (n=60) | 51 | **49/51 (96%)** |
| fabrication-prone (n=20) | 5 | **1/5 (20%)** |

The gate rejects most fabrication (15/20) — unstable guesses scatter under
sampling — but **five stable fabrications passed unanimously**. Consistency
measures conviction, not truth. This is why G2 licenses *hedged* speech only,
never assertion or training.

## E12b — anchored source trust in the memory (G1)

Three sources (real accuracies 95/80/30%) make conflicting claims; trust is
estimated from each source's record on a 30-fact verified anchor and used to
resolve contested entries.

| world | trust-weighted | fair majority |
| --- | --- | --- |
| clean | **98**/113 | 76/113 |
| best source compromised | **122**/171 | 68/171 |

The anchor detected the compromise (estimated trust 0.97 → 0.16) and
contained it; majority kept believing the fallen source. First-run lessons
kept on the record: a 20-sample anchor could not rank 95% vs 80% sources
(anchor size is a first-class parameter), and the v1 majority baseline was
accidentally biased by deterministic tie-breaking.

## E12c — is graded assertion calibrated?

The system answers a mixed set, labeling each reply with its licensing grade.

| grade | answered | accuracy |
| --- | --- | --- |
| G0 assert (oracle) | 30 | 100% |
| G3u assert (user-taught) | 30 | 100% |
| G2 hedged (consistent) | 49 | 78% |
| G4 declined | 11 | would have been 27% — declining was right |

Accuracy is strictly monotone in grade: the labels mean what they claim, so
assertion language ("verified" / "I believe" / "I don't know") can be driven
by metadata and stay honest.

## The finding across all three

External verification beats internal: the anchored gate (E12b) outperformed
its baselines and survived compromise; the internal gate (E12a) leaks
confident falsehoods. A drop of external truth (30 anchored facts) governs an
ocean of claims (300); zero drops govern nothing.

## Run

```powershell
python e12b_trust_store.py          # no GPU
cd ..\gpt
python ..\e12-verify\e12a_consistency.py out-epnat
python ..\e12-verify\e12c_calibration.py out-epnat
```
