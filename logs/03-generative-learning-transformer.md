# Generative Learning Transformer (fact memory slots on a frozen backbone)
**Dates:** 2026-08-19 to 2026-08-20 · **Status:** ongoing (session 4 planned; query-language Track A steps 5-7 and Track B not started) · **Hardware/scale:** single NVIDIA RTX 4080 SUPER (16 GB), Python 3.10, torch 2.11.0+cu128; backbone 3.21M parameters; memory up to 16,384 slots

## Question
Can a transformer whose backbone is frozen learn new facts at inference time through a separate slot memory, answer multi-hop questions over those facts, and retract a fact without damaging what else it knows? The motivation is a reported failure of knowledge editing: methods recall an injected fact but fail on multi-hop questions whose answers depend on it (about 15% to 34% on MQuAKE-CF in the cited band).

## Approach
- **Backbone:** a nanoGPT fork (`glt/model.py`, bit-identical to upstream logits before modification), 4 layers, 4 heads, 256 dim, 3.21M parameters. Trained from scratch on a synthetic universe of nonsense-word entities (200 entities) with relation paths rendered as nested "of" phrases. Phase 1 trains on 1,800 atomic facts and their compositions with 200 facts held out.
- **Memory:** `glt/memory.py`, a key-value slot table read in the forward pass. The final design addresses slots by a rolling hash of the content tokens of (relation, subject), not by hidden state.
- **Write protocol (`online.py`):** freeze the backbone and train only the slot table on the 200 held-out facts. Phase 2 has 1,048,576 trainable parameters, all in slots (2,048-slot runs). The slot count is 2,048 in the hidden-state runs (§14-§16); the token-address run size is not stated in §17, and `out/online.pt` has 16,384 slots.
- **Parser (`glt/language.py`, 0.89M parameters):** maps free English wording to a canonical token line, which is the slot address. Pointer head over input positions, 40 training wordings, held-out wordings and held-out content words scored separately.
- **Chaining and board (`chain.py`, `glt/workspace.py`, `ask.py`):** two-hop questions answered one hop at a time, each answer used as the next address. Provenance is recorded on a board outside the slots.
- **Evaluation sets:** 3,402 two-hop consequences of slot-written facts; 24 premises retracted one at a time with 200 control facts each (`cascade.py`).
- **Controls and baselines:** no-slot backbone (same architecture); all slots zeroed; memory switched off; hidden-state addressing versus token addressing; unconfined versus confined writes; single forward pass versus chained hops; wording and parameter/step sweeps for the parser.
- **Seeds:** the GLT runs are reported as single runs. The Llama-3-8B reference harness (Results row 1) pools seeds 0-3.
- **Pre-registered bars:** gates G1-G7 in `PROJECT_GOAL.md` §4 were set before the runs. G2 requires "high recall and minimal degradation, both, or it does not count." G3 asks to "substantially beat the ~15-34% band."

## Results
| Test | Result | Baseline / control | Verdict |
|---|---|---|---|
| Reference harness, Llama-3-8B-Instruct, facts injected in context (pooled seeds 0-3): open-book EM at 1 / 2 / 3 hops | 80.7% / 15.3% / 15.6% (forced choice 80.1% / 47.2% / 38.5%) | Closed book EM 0.0% at every depth; chance 25% | Harness calibration only; not a GLT result (in-context injection, not weight editing) |
| Backbone grokking, held-out composition (in-distribution / OOD) at 100,000 steps | 0.918 / 0.124 | 0.061 at 2,500 steps; chance 0.44% (1/225) | Positive: grokking reproduced |
| Slots added to backbone, no online writes (`slots_k1`, 2,048 slots, top_k = 1, 4.25M parameters): held-out composition | 0.918 | 0.918 without slots | Null: slots unused |
| Same model, all 2,048 slots zeroed: atomic recall | 0.9960 → 0.9960 | Memory on: 0.9960 | Null: slot values never moved from init |
| Online write, frozen backbone, unconfined write, 200 facts, 20,000 steps: online recall / kept atomic / online composition | 1.000 / 0.243 / 0.045 | Memory off: kept atomic 0.996 | Negative: fails G2 |
| Online write, confined to 200 fresh rows: online recall / kept atomic | 0.665 / 0.234 | Unconfined: 1.000 / 0.243 | Negative: confinement does not fix interference |
| Read schemes, unconfined write: softmax / gated / threshold (online recall / kept atomic) | 1.000 / 0.243; 0.570 / 0.202; 0.315 / 0.091 | Softmax as default | Negative: none fixes kept-fact damage |
| Hidden-state address separation (cosine, online vs. nearest online / kept vs. nearest online) | 0.8836 / 0.8486 (margin 0.0351) | n/a | Negative: addresses not separable |
| Token-hash address, frozen backbone, 20,000 steps: online recall (0 → 20,000) | 0.005 → 1.000 | Hidden-state address: 1776 of 1800 kept facts colliding (98.70%) | Positive: G2 passes (both halves) |
| Token-hash address: kept atomic (0 → 20,000) | 1.000 → 1.000 | Hidden-state address: 0.243 | Positive: G2 passes |
| Token-hash address: online composition (0 → 20,000) | 0.009 → 0.009 | Backbone composition: 0.918 | Negative: G3 fails |
| Token-hash model, atomic recall with all slots zeroed | 0.0095 | Memory on: 1.0000 (hidden-state address: 0.0000 drop) | Positive: atomic facts stored in slots |
| Sequential writes on token-hash model | 2,000 prior facts at 1.000; zero drift over 100 sequential writes | n/a | Positive (as summarised in PROJECT_GOAL §18) |
| Single forward pass, two-hop consequences (3,402) | 0.010 | Symbolic chain, decomposition supplied: 1.000 | Negative for single pass; positive for chaining |
| Free-form English end to end (`ask.py`, 3,402 questions, distractor name planted) | 1.000 read correctly; 1.000 answered | Single forward pass: 0.010 | Positive (40 wordings, closed vocabulary) |
| Retraction cascade (`cascade.py`): premise recalled after slot zeroed | 0.000 | Backbone: no retraction (§17) | Positive |
| Retraction cascade: consequences withdrawn / collateral damage / provenance predicted | 0.995 / 0.000 / 0.995 | 200 controls per retraction | Positive, with caveat (see Limitations) |
| Recorded provenance vs. ground-truth entailment graph | 3,402 of 3,402 | n/a | Positive |
| Consequences keeping the same answer word via a different intermediate | 0.005 | Old derivation gone in all cases | Reported separately from withdrawal |
| Parser, held-out content words, 1-hop: generate head / pointer head | 0.000 / 0.438 | Trained words: 1.000 for both | Positive for pointer over generate |
| Parser, held-out wordings, 1-hop / 2-hop: generate / pointer | 0.600 / 0.895 (1-hop); 0.762 / 0.887 (2-hop) | 32 training wordings | Positive for pointer |
| Parser, parameter sweep 0.14M → 14.88M (1-hop / 2-hop, held-out wording) | 0.522 → 0.490 / 0.625 → 0.500 | 8 wordings, 4k steps | Null |
| Parser, step sweep 4,000 → 100,000 (1-hop / 2-hop, held-out wording) | 0.565 → 0.527 / 0.468 → 0.398 | 8 wordings, 0.89M | Null |
| Parser, wording sweep 4 → 32 (1-hop / 2-hop, held-out wording) | 0.445 → 0.895 / 0.075 → 0.887 | 0.89M, 4k steps | Positive: wording diversity is the lever |
| Parser invariance, 80 phrasings of 40 queries (1-hop and 2-hop) | 1.000 agree; 1.000 agree and correct | Held-out wordings and words together: 0.168 / 0.000 | Positive, narrow |
| Query operations, exact set or value match: follow / reverse / count / same / none | 1.000 / 0.970 / 0.970 / 1.000 / 1.000 | reverse before confidence floor: 0.660 | Mixed: reverse and count bounded by scan error |
| Reverse scan: invented names per true edge, no floor → floor 0.99 | 0.401 → 0.032 | Floor sits below all true edges (min 0.998) | Positive, but not exact |
| Query-operation parser after adding the query shapes: at 4,000 steps, invariance 0.990 and held-out wordings 0.792 (was 0.895); at 12,000 steps, 1.000 on every shape and 1.000 invariance at both depths | 0.792 → 1.000 (held-out wordings / shapes at 12,000) | 0.895 held-out wordings before the wider query language | Positive: wider query language costs training steps at fixed capacity |

## What it showed
- **Facts can be written into slots at inference, with the backbone frozen, but only when the address is exact.** With hidden-state addressing, 98.70% of kept facts collided with written slots and kept-fact accuracy fell to 0.243 (memory on) versus 0.996 (memory off). Switching to a token-hash address removed the collisions, gave online recall 1.000, and kept atomic accuracy at 1.000.
- **Deletion by zeroing a slot worked in the tested cases.** Premise recall fell to 0.000, and 0.995 of derived consequences were withdrawn with 0.000 collateral across 24 retractions, using a provenance board outside the slots. This is tested on a closed synthetic universe with a single run.
- **Multi-hop answers from slot-stored facts failed inside the network and succeeded only when the question was split into hops.** A single forward pass scored 0.010 on 3,402 two-hop consequences, and online composition stayed at 0.009 after 20,000 steps. A symbolic chain using the same memory scored 1.000, but the chainer was given the decomposition, so the 1.000 reflects the chainer and not the transformer.
- **Generation of unseen words failed; pointing worked.** A generative head scored 0.000 on held-out content words. A pointer head over input positions scored 0.438 on the same words and 0.895 on held-out wordings. Wording diversity moved held-out accuracy; parameters (0.14M to 14.88M) and steps (4k to 100k) did not.
- **Several things remain open or negative.** Reverse lookup and counting are bounded by confabulation during scans (0.401 invented names per true edge before the floor, 0.032 after). Abstention (G4) is not built. Gradient-trained slot writes with unconfined updates damaged kept facts and softmax, gated and threshold reads all failed to fix it.

## Limitations
- **Scale and data:** 200 nonsense-word entities, closed vocabulary, synthetic relation graph. Two-hop paths are the deepest tested for the online experiments.
- **Seeds:** most GLT results are single runs with no seed count given. The project notes treat single-seed Llama numbers as noise (§12) and state that G5b's 120 probe trials (2 events) lack the power to gate anything.
- **Language:** the parser covers 40 project-written wordings. Held-out wordings and words together scored 0.168 (1-hop, per the table in §19). Invariance restricted to the 8 held-out wordings was 0.000 at every sweep point, so per-phrasing errors are correlated. Real English has not been tested (Track B not started).
- **Gates not measured:** G5b (adversarial recovery) and G5c (competition margin) have not been re-measured on the slot-and-board system. G4 (abstention) is not built. G6 (arithmetic) and G7 (decision) are not started.
- **Cascade caveat:** the retraction test has 24 premises and depth-2 paths only. The project notes that a zeroed slot plus a discarded board is a prior, not a measurement, for adversarial recovery.
- **Reverse and count:** `reverse` scans every entity (cost grows with entity count), and its exact-match score of 0.970 cannot be pushed to 1.000 by filtering.
- **Comparison baseline:** the Llama-3-8B harness injects facts in context. Its multi-hop numbers match the magnitude of the MQuAKE-CF band but are not a same-method replication, which the project notes is a weight-editing setting.
- **Persistence:** the board is per session. Provenance is not persisted across restarts, and rules are not implemented.
- **Documentation conflicts:** the handoff gives about 25 gradient steps per fact write, while PROJECT_GOAL §18 gives about 10. The test count differs between HANDOFF (183) and PROJECT_GOAL §22 (202). Neither number is used in the Results table.

## Key files
- `glt/model.py`: nanoGPT fork with slot read and the token-hash `address()`; bit-identical to upstream when slots are off.
- `glt/memory.py`: `MemorySlots`; token-address read used in the final design.
- `online.py`: two-phase write experiment (frozen backbone, slot-only phase 2) that produced the G2 and G3 numbers.
- `glt/universe.py`, `glt/questions.py`, `glt/corpus.py`: synthetic entity/relation universe, entailment closure with recorded premises, question generation, and the ID/OOD split.
- `glt/probes.py`, `glt/evaluate.py`: retraction probes and the Llama-3-8B reference harness (G1 perplexity, exact-match and forced-choice scorers).
- `chain.py`: two-hop chaining over the memory, with the decomposition supplied by the harness.
- `glt/language.py`: the parser (pointer head, canonical token output).
- `extract.py`, `capacity.py`: parser wording, parameter and step sweeps.
- `glt/workspace.py`: the board, control loop, provenance and retraction; no torch dependency.
- `ask.py`: free-form English end-to-end evaluation on the 3,402 questions.
- `cascade.py`: G5a retraction measurement (bookkeeping and substrate checks).
- `operations.py`: `reverse`, `count`, `same` against ground truth, with the confidence floor.
- `chat.py`: interactive demo (teach, forget, where, bits).
- `out/online.pt`: the token-addressed model with 16,384 slots and 200 facts written; `out/parser.pt`: the deployed parser.

## Sources
- `HANDOFF.md`: session 4 handoff; sections 1-3 (results table, design decisions, open questions), 5 (run commands and files), 7 (dead ends).
- `PROJECT_GOAL.md`: §4 (gates G1-G7), §12 (Llama harness numbers, G5), §13 (grokking run), §14 (slot read path, 1,048,576 phase-2 parameters), §15 (online writes, confined vs. unconfined, read schemes), §16 (cosine margin), §17 (token addressing, 98.70% collisions, atomic zeroing), §18 (chaining, 0.010 vs. 1.000), §19 (parser: failures, sweeps, invariance gate), §20 (end-to-end English), §21 (G5a cascade), §22 (query operations, floor, contradiction).
- `THEORY.md`: §9 (interpretation: addressing by structure, exactness bounded by memory reads); numbers are cross-checked against `PROJECT_GOAL.md`.
