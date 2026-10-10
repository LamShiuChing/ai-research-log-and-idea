# Continual learning falsification testbed (re-testing Dohare et al., Nature 2024, loss of plasticity)
**Dates:** 2026-09-01 (round 1; latest dated entries in the docs are 2026-09-03) · **Status:** ongoing (round 1 complete; claim C10 undecided; later rounds moved to language models) · **Hardware/scale:** round 1 on an RTX 4080 (README: "minutes each on a 4080"), torch 2.11 + cu128; MNIST only, scaled down from the paper

## Question
Does loss of plasticity in deep continual learning reproduce on a scaled-down online permuted-MNIST stream? Do the remedies the paper names (L2, shrink-and-perturb, continual backprop) help, and does the stronger claim that catastrophic forgetting is "totally curable" with the ideas behind continual backprop hold? Each claim was written with the observation that would refute it before its number was taken.

## Approach
- **Model:** ReLU MLP, 3 hidden layers of 256 units (the paper uses 2000). Single head, no task label.
- **Data:** online permuted MNIST. Each task is a fixed pixel permutation; every training example is seen once, in random order. Online accuracy is measured before the update on each example.
  - Plasticity claims (C1–C6, C10, C11): 200 tasks × 10,000 examples. Paper: 800 tasks × 60,000.
  - Forgetting claims (C7–C9, C8b): 10 tasks × 60,000.
- **Arms:** plain backprop (SGD, lr 0.05, batch 16); Adam (lr 1e-3); L2 (weight decay 1e-4); shrink and perturb (decay 1e-4, noise 3e-5, and 1e-4 for C4b); continual backprop (replacement rate 1e-4, maturity 100, decay 0.99); CBP with random unit replacement (C11); per-task norm reset (C10); replay reservoir of 2,000 mixed 1:1 into each batch (C9).
- **Controls:** seed 0 as a pilot that set the scale (never a verdict); random reinitialisation (C11); a second noise scale (C4b); a second CBP rate (C8b); the no-replay baseline for forgetting (C7).
- **Seeds:** verdicts on seeds 1, 2, 3. A claim survives if it holds on at least 2 of 3. Results within 2σ of a bar escalate to seeds 4–9 under the amended conventions.
- **Bars (written before the verdict runs):** plasticity claims use a 1-point margin on the last-window vs best-window drop; forgetting claims use a 2-point margin (10 points for the naive-training control C7).
- **Hyperparameters:** taken from the paper's grid, scaled, and recorded in the ledger before running. None were tuned on the verdict seeds.

## Results
Online accuracy is the mean over the last 10 % of tasks (last window). "Best − last" is the best sliding window minus the last window. Forgetting is the mean over earlier tasks of (best-ever − final) test accuracy. Values are seeds 1 / 2 / 3, copied from the Evidence section of `CLAIMS.md`.

| Claim | Result | Baseline / control | Verdict |
|---|---|---|---|
| C1 Plain backprop loses plasticity | BP last 0.828 / 0.827 / 0.831; best − last 0.043 / 0.042 / 0.039 | Own best window | SURVIVES 3/3 |
| C2 Adam is worse | Adam last 0.810 / 0.815 / 0.817 | BP last 0.828 / 0.827 / 0.831 (Adam below BP on each seed) | SURVIVES 3/3 |
| C3 L2 mitigates | L2 last 0.844 / 0.841 / 0.847; best − last 0.027 / 0.030 / 0.024 | BP last 0.828 / 0.827 / 0.831 | SURVIVES 3/3; mitigates but does not eliminate the drop |
| C4 Shrink and perturb (noise 3e-5) nearly eliminates the loss | Last 0.837 / 0.845 / 0.844; best − last 0.034 / 0.025 / 0.027 (values from the Evidence row labelled "snp 1e-4 / 3e-5") | Bar: within 1 point of own best; no better than L2 | FALSIFIED 3/3 |
| C4b C4 is not a noise-scale artefact (noise 1e-4) | Last 0.840 / 0.844 / 0.847; best − last 0.031 / 0.027 / 0.024 | L2 last 0.844 / 0.841 / 0.847 | FALSIFIED 3/3 |
| C5 Continual backprop maintains plasticity | CBP last 0.865 / 0.867 / 0.869; best − last 0.009 / 0.005 / 0.005 | BP last 0.828 / 0.827 / 0.831 | SURVIVES 3/3 on the BP comparison; the own-drop part passes by 0.1 point on seed 1 (survives with caveat) |
| C6 Dead units, weight size and effective rank move as the paper describes | BP, seed 1 only: dead units 0.027 → 0.146; mean \|w\| 0.031 → 0.073; effective rank 41 → 51 | Prediction: rank falls | FALSIFIED 3/3 (rank rises under SGD; it falls only under Adam) |
| C7 Naive training forgets catastrophically (control) | BP forgetting 0.390 / 0.355 / 0.342 | Bar ≥ 0.10 | SURVIVES |
| C8 CBP alone cuts forgetting by ≥ 2 points ("totally curable") | CBP forgetting 0.640 / 0.680 / 0.660 | BP forgetting 0.390 / 0.355 / 0.342 | FALSIFIED 3/3 (CBP forgets more) |
| C8b C8 is not a replacement-rate artefact (rate 1e-5) | CBP forgetting 0.435 / 0.466 / 0.433; final 0.571 / 0.543 / 0.570 | BP forgetting 0.390 / 0.355 / 0.342 | SURVIVES 3/3 (CBP still 4–11 points worse than BP) |
| C9 Replay reduces forgetting | Replay forgetting 0.024 / 0.023 / 0.021; final 0.943 / 0.944 / 0.944 | BP forgetting 0.390 / 0.355 / 0.342 | SURVIVES |
| C10 Weight-norm growth alone explains the loss | Norm-reset last 0.845 / 0.851 / 0.856; best − last 0.019 / 0.014 / 0.007 | BP and L2 last windows as above; bar best − last ≤ 0.01 | UNDECIDED (n = 3). The registered rule would falsify on seeds 1 and 2; escalation to seeds 4–9 was required and has not been run |
| C11 CBP's utility measure adds nothing over random reinitialisation | Random-reinit last 0.864 / 0.863 / 0.865 | CBP last 0.865 / 0.867 / 0.869 (differences 0.1 / 0.4 / 0.4 points) | SURVIVES 3/3 |

Later rounds (C13 onward, from `CLAIMS.md`; summary only):
- **Coverage and bounded memory:** C13a survives 3/3 (a threshold, not a slope); C13b and C13c falsified after escalation (0/9 and 3/9).
- **Sleep and merging (C15):** C15a survives 9/9 after escalation; C15d (weight-addition merge) survives as a failure claim (S_ret 72.6 / 62.7 / 64.9 points); C15f (random reinit inside sleep is safe) falsified 3/3; C15j falsified 5/9.
- **Routing (C16):** C16a falsified 3/3; C16b falsified 3/3, with always-continue within 1.0–1.7 points of the doorman.
- **Composition (C21):** C21a, C21b and C21c all falsified.
- **Retention economics (C22, C22b, C25):** C22 falsified 3/3 on the MNIST world (savings +0.1 / −0.1 / +0.2 %). C22b survives 3/3 (sparser recurrence, savings +1.6 / +0.9 / +1.1 %, still under the 10 % bar). C25a survives 3/3 on a substrate where relearning is expensive (savings 25.2 / 24.9 / 25.5 %). C25b falsified 3/3.
- **Language substrate (C31–C33):** C31 and C32a survive; C32c (weight-addition merge without rehearsal) falsified 3/3 (S_ret 2.180 / 2.154 / 1.949 bpc against a 0.20 bar); C33a survives 2/2; C33b, C33c and C33d falsified 2/2.

## What it showed
- Plain backprop on this stream shows the paper's first effect: the last-window accuracy is 3.9–4.3 points below its best window (C1), and Adam is worse on every seed (C2). This is measured at one hyperparameter setting per arm.
- L2 reduces the drop but does not remove it (C3). Shrink and perturb did not beat L2 at either noise scale tested (C4, C4b).
- The utility-based selection in continual backprop added nothing over choosing units at random (C11). Its replacement step is what does the work.
- Continual backprop increases forgetting: it reaches 0.640–0.680 forgetting against 0.342–0.390 for plain backprop (C8). Reinitialising units removes information. Replay was the only retention mechanism tested that clearly reduced forgetting (C9).
- The mechanism in the paper is not fully confirmed. Under SGD, effective rank rises (C6), and the rank collapse appears only under Adam. The claim that weight growth alone explains the loss (C10) is undecided, not confirmed.

## Limitations
- Scaled down from the paper: batch 16 (paper: 1), 256 units per layer (paper: 2000), 200 tasks × 10,000 examples for plasticity (paper: 800 × 60,000). MNIST only. No language model in round 1.
- One hyperparameter setting per arm, taken from the paper's grid and scaled. Falsifications hold at these settings, not in general. Adam uses lr 1e-3 rather than the paper's grid.
- CBP is implemented from the paper's equation 1 and omits the downstream bias compensation in the authors' code. The result is for this variant.
- Three verdict seeds. C10 is undecided and C5's drop component is a near-bar pass. Seeds 4–9 were not run for either.
- The numbers here are copied from the Evidence section of `CLAIMS.md`, not recomputed from `results/`, which is git-ignored and regenerable.
- Ledger inconsistencies, flagged rather than resolved:
  - C8's text says CBP raises forgetting by "27–33 points". The table values give differences of 25.0, 32.5 and 31.8 points. The table values are used above.
  - C5's text says the margin over BP is "3.6–3.8 pts". The table values give 3.7, 4.0 and 3.8 points. The table values are used above.
  - The ledger header says "7 survive, 6 falsified". The per-claim verdicts give 8 survive (C1, C2, C3, C5, C7, C8b, C9, C11), 4 falsified (C4, C4b, C6, C8) and 1 undecided (C10).
  - The Evidence section says C10 was "FALSIFIED 2/3". That text predates the methodology re-grade, which set C10 to undecided. The later status is used.
- Sources S2 (podcast) and S3 (workshop paper) were not fully retrievable. S2 is used only for its blurb. S3's wording is secondary and unverified. S4 was not read and is used only as a generic frame.

## Key files
- `README.md`: project aim, run commands, layout, and the four rules (falsifier first, seed 0 pilots, one variable per arm, nulls recorded).
- `CLAIMS.md`: the claim ledger. Conventions, sources S1–S4, the testbed deviation table, every claim C1–C37 with its falsifier and status, and the Evidence section with the round-1 numbers.
- `THESIS.md`: the argument that the round-1 results were meant to test; it is read as context, not as evidence.
- `run.py`: trains one learner on the online permuted-MNIST stream and writes per-task online accuracy, the test-accuracy matrix, and plasticity diagnostics to JSON. Flags select the arm.
- `checks/plasticity.py`: runs the eight plasticity arms on seeds 1–3 and prints one verdict per claim (C1–C6, C10, C11). Defines the bars.
- `checks/forgetting.py`: runs the forgetting arms (C7–C9, C8b).
- `checks/lib.py`: seeds (1, 2, 3), the windowing function, and the 2-of-3 verdict rule.
- `cl/cbp.py`: continual backprop (generate-and-test on hidden units).
- `cl/learners.py`: one online update covering SGD or Adam, L2, noise, CBP and replay.
- `cl/net.py`, `cl/data.py`, `cl/metrics.py`: the MLP, the permuted-task stream, and the accuracy and diagnostic metrics.
- `notes/review-methodology.md`: the methodology review that re-graded C10 and C5 and added the 2σ escalation rule.

## Sources
- `README.md`: aim, run commands, hardware note, limits.
- `CLAIMS.md`: conventions, sources S1–S4, testbed deviations, claim statements and falsifiers (C1–C11, C13 onward), Evidence section (all round-1 numbers in the Results table, and the C13+ verdicts).
- `checks/plasticity.py`: arm definitions and the verdict rule; the bars in the code match the claim statements.
- `checks/lib.py`: seed set and verdict rule.
- `THESIS.md`: later-round verdicts on the retention and economics claims (C31 savings range).
- `NEXT_SESSION.md`: confirms C10 is still undecided and that the seeds 4–9 escalation is open.
- `HOLES.md` (E23): confirms the C10 and C5 re-grade.
- `notes/review-methodology.md`: the basis for the C10 and C5 re-grade.
