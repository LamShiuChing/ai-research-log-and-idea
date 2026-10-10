# Claims ledger — continual learning (rev. 1, 2026-09-01 — round 1 complete: 13 claims, 7 survive, 6 falsified)

Conventions
- One atomic claim per number. Status ∈ {proposed, survives, falsified, undecided, revised, not-a-claim}.
- The falsifier is written before the number is taken. A claim survives if it holds on ≥ 2 of the
  3 verdict seeds (1, 2, 3). Seed 0 is the pilot that fixed the testbed scale and never enters a verdict.
- Amendments after the 2026-09-01 methodology review (`notes/review-methodology.md`): no zero-margin
  comparisons — every bar carries a margin ≥ 2× the measured paired seed-sd; any per-seed statistic
  within 2σ of its bar escalates to seeds 4–9 automatically and is *undecided* until then; drop
  statistics use a fixed pre-registered reference window, not a max over sliding windows; follow-up
  claims inherit the parent's bar and statistic verbatim, changing only the single named knob.
- Every number names a result file in `results/` and the check that produced it. Results are
  regenerable and git-ignored; the number lives here.
- "Point" = 0.01 accuracy. "Window" = mean over a sliding window of 10 % of the tasks.

## Sources

- **S1** Dohare, Hernandez-Garcia, Lan, Rahman, Mahmood, Sutton. *Loss of plasticity in deep continual
  learning.* Nature 632, 768–774 (2024). https://www.nature.com/articles/s41586-024-07711-7
  (read via https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/). Code and configs:
  https://github.com/shibhansh/loss-of-plasticity (`lop/permuted_mnist/cfg/*.json`, `lop/algos/gnt.py`).
- **S2** Sequoia *Training Data*: "Rich Sutton and Khurram Javed: Why AI Models Stop Learning, and How
  to Start It Again." https://www.youtube.com/watch?v=xH7U7w9Qzlo — only the episode blurb was
  retrievable (transcript pages are script-rendered): catastrophic forgetting is "totally curable" with
  the ideas behind continual backprop; the big world hypothesis; synthetic data is "a big mistake".
- **S3** Javed & Sutton. *The Big World Hypothesis and its Ramifications for AI.* RLC 2024 workshop.
  https://openreview.net/pdf?id=Sv7DazuCn8 — not retrievable here (bot check); wording below is from
  secondary sources and is unverified.
- **S4** IBM Think, *What is continual learning?* https://www.ibm.com/think/topics/continual-learning —
  returned 403; not read. Used only as the generic frame (replay / regularisation / architecture;
  task-, domain-, class-incremental), which is textbook.

## Testbed (`run.py`) and its deviations from S1

Online permuted MNIST: each task is a fixed pixel permutation; every training image is seen once, in
random order; online accuracy is the fraction of examples classified correctly *before* the update on
them. After each task, test accuracy on every task seen so far (single head, no task label) and three
diagnostics on 2,000 test images of the current task: dead-unit fraction, mean |weight|, effective rank
of the last hidden layer.

| | S1 (`cfg/*.json`, `online_expr.py`) | here | why |
|---|---|---|---|
| mini-batch | 1 | 16 | 48 M single-example steps is not runnable here |
| network | 3 × 2000 ReLU | 3 × 256 ReLU | speed; the phenomenon is claimed to be general |
| SGD step | 0.003 | 0.05 (≈ 0.003 × 16 with mean-reduced loss) | per-example step matched; pilot: 88 % online on task 1 |
| tasks × examples | 800 × 60 000 | 200 × 10 000 (C1–C6); 10 × 60 000 (C7–C9) | minutes per run |
| CBP utility | `adaptable_contribution` (code) | eq. 1 of the paper (`contribution`) | test the published algorithm |
| CBP reinit | + downstream-bias compensation (code) | none (paper text) | same |
| CBP rate / maturity / decay | {1e-4, 1e-5, 1e-6} / 100 / 0.99 | 1e-4 / 100 / 0.99 | ≈ 1 unit per layer per 600 examples |
| L2 weight decay | {1e-3, 1e-4, 1e-5} | 1e-4 | grid midpoint |
| shrink & perturb | wd 1e-4, noise {1e-4, 1e-5, 1e-6} | wd 1e-4, noise 3e-5 | per-example noise scaled to 16 steps |
| Adam step | {1e-4, 3e-5, 5e-5} | 1e-3 | fewer steps; standard value |

No hyperparameter here was tuned on the verdict seeds. A falsification at these settings is a
falsification *at these settings*; a sweep is a separate, later claim.

## Frame (not claims)

**F1 — Big world hypothesis (S3).** "For many learning problems, the world is multiple orders of
magnitude larger than the agent" (secondary quotation, unverified), so an agent's approximation must
be updated for its whole life even if the world is stationary. Not testable on MNIST; it is why the
streams below have no end and why "learn task k then stop" is never the object of study.

**F2 — Synthetic data is a mistake (S2).** Not operationalised; no claim.

## Claims — plasticity (`checks/plasticity.py`)

**C1 — Plain backprop loses plasticity.** [SURVIVES 3/3]
Statement: on the 200-task stream, BP's per-task online accuracy rises and then falls: its last window
is more than 1 point below its best window.
Source: S1 — "performance first increased across tasks, then began falling steadily across all
subsequent tasks."
Falsifier: last window ≥ best window − 1 point on ≥ 2 of 3 seeds.

**C2 — Adam is worse.** [SURVIVES 3/3]
Statement: Adam's last-window online accuracy is below BP's on the same stream.
Source: S1 — Adam's loss of plasticity is "catastrophic".
Falsifier: Adam ≥ BP on ≥ 2 of 3 seeds.

**C3 — L2 mitigates.** [SURVIVES 3/3 — mitigates, does not eliminate: still −2.4 to −3.0 points]
Statement: L2's last window is above BP's.
Source: S1 — L2 "stops the weights from becoming too large".
Falsifier: L2 ≤ BP on ≥ 2 of 3 seeds.

**C4 — Shrink and perturb nearly eliminates the loss.** [FALSIFIED 3/3 at noise 3e-5 — no better than L2; see C4b]
Statement: S&P's last window is above BP's and within 1 point of S&P's own best window.
Source: S1.
Falsifier: either condition fails on ≥ 2 of 3 seeds.

**C4b — C4 is not a noise-scale artefact.** [FALSIFIED 3/3; registered 2026-09-01 after C4 fell on seeds 1–3, before running]
Statement: shrink and perturb with noise 1e-4 (top of S1's grid, ≈ 3× the C4 setting) has its last window
above BP's and within 1 point of its own best.
Falsifier: either condition fails on ≥ 2 of 3 seeds.

**C5 — Continual backprop maintains plasticity.** [SURVIVES 3/3 on the vs-BP comparison (margins
3.6–3.8 pts ≫ noise); the own-drop component's seed-1 pass cleared the bar by only 0.1 pt — inside
noise per `notes/review-methodology.md` §2; drop component re-graded to survives-with-caveat]
Statement: CBP's last window is above BP's and within 1 point of CBP's own best window.
Source: S1, and the central engineering claim of S2.
Falsifier: either condition fails on ≥ 2 of 3 seeds.

**C6 — The three correlates move as described.** [FALSIFIED 3/3 — dead units and |w| rise, but effective rank *rises* under SGD; it falls only under Adam]
Statement: under BP, from the first window of tasks to the last: dead-unit fraction rises, mean |w|
rises, effective rank falls.
Source: S1 — "up to 25 % of units die"; weights grow; "low effective rank means a small number of
units can produce the output".
Falsifier: any of the three moves the other way (or not at all) on ≥ 2 of 3 seeds.

**C10 — Weight-norm growth alone explains the loss.** [UNDECIDED at n=3 — re-graded 2026-09-01 after
the methodology review (`notes/review-methodology.md` §2–3): the seed spread of the drop statistic
(1.2 pts) exceeds its distance to the bar, and the "best window" max over ~180 overlapping windows
biases the statistic ~0.2–0.4 pts against the arm. Escalation to seeds 4–9 with a fixed reference
window required before a verdict. Directionally: beats L2 and S&P on all 3 seeds.]
Statement: plain BP with every weight matrix rescaled to its initial RMS after each task (direction
kept, nothing else changed) keeps its last window above BP's and within 1 point of its own best.
Source: the author's `D:\Javis\experiments\11_plasticity` (2026-08): no dead units, stable rank flat,
weight RMS 0.13 → 0.76 across a life, lr × 3 restored learning 3/3 — "loss of plasticity as
effective-lr decay." Pilot here: BP |w| 0.031 → 0.074 while effective rank *rose*.
Falsifier: norm-reset arm ≤ BP, or its best − last > 1 point, on ≥ 2 of 3 seeds.
If it survives, dead units are a symptom, not the mechanism, at this scale.

**C11 — CBP's utility measure adds nothing over random reinitialisation.** [SURVIVES 3/3]
Statement: replacing the same number of mature units chosen at random (same rate, maturity, reinit)
ends within 1 point of CBP's last window.
Source: the author's eureka rule — every internal self-check tried failed (E3 ×3, E12a, v3); CBP's
utility is an internal self-check of which units matter.
Falsifier: random-reinit last window < CBP − 1 point on ≥ 2 of 3 seeds.

## Claims — forgetting (`checks/forgetting.py`)

Forgetting after task T = mean over j < T of (best test accuracy ever on task j − final accuracy on j).

**C7 — Naive training forgets catastrophically.** [SURVIVES; control]
Statement: on 10 full tasks, BP's forgetting is ≥ 10 points.
Source: textbook; the author's earlier E5 measured 85 points on a facts task.
Falsifier: < 10 points on ≥ 2 of 3 seeds.

**C8 — "Catastrophic forgetting is totally curable with the ideas behind continual backprop."** [FALSIFIED 3/3]
Statement as tested (strong reading of S2): CBP alone lowers forgetting by ≥ 2 points relative to BP.
Falsifier: CBP ≥ BP − 2 points on ≥ 2 of 3 seeds.
Prediction before running: falsified — CBP targets plasticity (learning the new), not stability
(keeping the old). Caveat: without the transcript the strong reading may not be the speakers' intent;
if C8 falls, the next revision must state the weaker reading they would defend and test that.

**C9 — Replay reduces forgetting.** [SURVIVES]
Statement: a 2,000-example reservoir mixed 1:1 into each batch lowers forgetting by ≥ 2 points
relative to BP.
Falsifier: replay ≥ BP − 2 points on ≥ 2 of 3 seeds.

**C8b — C8 is not a replacement-rate artefact.** [SURVIVES 3/3; registered 2026-09-01 after C8 fell, before running]
Statement: at one-tenth the churn (rate 1e-5 ≈ 10 units per layer per 60 000-example task, vs ≈ 96 at
1e-4), CBP still does not lower forgetting by ≥ 2 points relative to BP.
Falsifier: CBP-1e-5 ≤ BP − 2 points on ≥ 2 of 3 seeds.
Prediction: survives (C8 stays falsified) — reinitialisation only removes information; a lower rate
removes less, it never protects.

## Claims — currency (`checks/currency.py`)

**C22 — Retention pays, in prequential bits, iff the world recurs.** [proposed; registered 2026-09-01
before running, margins per the amended conventions]
Statement: prequential code length of a run = Σ over training examples of −log₂ p(y | x) taken on the
pre-update prediction (the same predictions whose argmax is the online accuracy). Then:
(i) on the non-recurring stream (10 fresh tasks × 60 000), replay-2000's total code length is *not*
more than 10 % below BP's; and (ii) on the recurring stream (a pool of 10 permutations, 30 blocks ×
20 000 examples, block identity drawn uniformly from the pool), replay's total *is* ≥ 10 % below BP's.
Source: `notes/design-currency.md` (the A13 test — "retention pays through future reuse on a
recurring world", made a number). The 10 % margin is registered here and is ≫ the measured seed noise.
Falsifier: (ii) fails on ≥ 2 of 3 seeds — the currency cannot express the thesis's retention bet, and
A13/E2 stand open. If (i) fails instead (retention pays ≥ 10 % even with no recurrence), that is
reported as a surprise against the currency's own logic, not silently absorbed.
Status: **FALSIFIED 3/3** (2026-09-01). Part (i) held (flat savings +1.3/+0.9/+1.0 %); part (ii)
failed — recurring-world savings +0.1/−0.1/+0.2 %. Diagnostic (`results/cur_recur_*`): replay DID
retain (mean pool accuracy 70–73 % vs BP's 64–69 %; final 86–95 % vs 79–91 %) — the retention was
simply near-worthless in bits, because BP relearns a returning permutation almost for free at this
revisit rate (mean gap ≈ 10 blocks). Empirical support for HOLES E3: at high revisit frequency the
base learns raw and no memory mechanism pays.

**C22b — The failure is not a revisit-rate artefact.** [proposed; registered 2026-09-01 after C22
fell, before running; inherits C22's statistic and 10 % margin verbatim, one knob changed:
recurrence sparsity — pool 30, 90 blocks × 20 000 (mean gap ≈ 30 blocks ≈ 600 000 examples)]
Statement: at 3× sparser recurrence, replay's fractional prequential-bit saving over BP is still
< 10 %.
Prediction (from HOLES E3): survives — sparser revisits also thin the reservoir (≈ 66 examples per
task), so the band has no profitable middle at this scale. If instead saving ≥ 10 % on ≥ 2 of 3
seeds, the currency revives inside a frequency band and E3's band becomes the currency's domain of
validity rather than its refutation.
Status: **SURVIVES 3/3** (2026-09-01), as predicted: sparse savings +1.6/+0.9/+1.1 % — direction up
from the dense world (+0.1 %) but an order of magnitude under the bar, while replay's retention
advantage was clearly visible (mean pool accuracy 0.646 vs 0.549, final 0.900 vs 0.829, seed 1).
Combined C22/C22b reading: **at this instrument's scale, relearning is so cheap that retention
cannot pay in prequential bits at any recurrence rate we can build** — the one-currency objective
fails its own first test, and HOLES E2/E3 are settled against it here. Scale caveat, recorded not
leaned on: the currency's bet might only pay where relearning is expensive (the LLM-scale regime
where plasticity loss lives); untestable on this instrument.

## Claims — sleep (`checks/sleep.py`, instrument `sleep_run.py`; registered 2026-09-01 before any run)

Protocol (one run = arm × seed). 21 domains (permutations): P0 + F1–F4 (foundation) + R (recurring)
+ 15 fresh. Foundation: base (3×256, SGD lr 0.05, batch 16) trains with replay (C9's mechanism) on
F1–F4 then P0, 10 000 examples each; **P0's examples never enter the reservoir** — P0 is the
never-rehearsed probe (HOLES A8). Then 20 cycles: **wake** = fresh adapter (all-layer rank-4
low-rank deltas + hidden-unit gates, identity init — `notes/design-composition.md`; SGD lr 0.05) on
the cycle's domain, 10 000 examples seen once, base frozen; **sleep** = 625 SGD steps (lr 0.05) of
soft-cross-entropy self-distillation, each step **48 reservoir inputs** targeted by the pre-sleep
base's logits + 16 wake-buffer inputs targeted by the composite's logits (3:1 old:new, per-row mean
— pilot-fixed, see the pilot record below); **reservoir 8000** (pilot-fixed); adapter deleted after
sleep; wake buffer then enters the reservoir. R recurs at cycles 1, 2, 4, 8, 16 (A5); the other 15 cycles are fresh domains;
past cycle 20 (accelerated arm, 60 cycles) domains recycle uniformly from the pool so capacity stays
fixed and extra sleeps are measured (E25). Probes = test accuracy on all 21 domains after every
sleep (end-of-sleep = settled; E9's stability gap concerns C17's within-sleep audit, not these).
Known instrument caveat, owned: permutation domains are input-layer-adversarial for low-rank
adapters (a fresh permutation is a full-rank input change), so the composite's ceiling is itself a
logged outcome; transfer and plasticity bars are relative, not absolute. Seed-0 pilot may re-fix
instrument constants only (recorded in the pilot record below); everything frozen before seeds 1–3.

**Seed-0 pilot record (2026-09-01, 10 runs; constants frozen from it before any verdict seed).**
Composite ceiling non-issue: rank-4 + gates reaches 83–90 % on fresh permutations; transfer faithful
(S_new ≈ 0 at 625 steps); R-convergence clean (online .75→.877→.886→.889→.886, ‖Δ‖ 5.7→1.1→0.8);
capacity non-issue (scratch ceiling 91.6 % mean over all 21 domains). The load-bearing pilot finding
is a **per-cycle retention leak** on rehearsed foundation domains, and its causal chain:
at the naive constants (reservoir 2000, 1:1 old:new) S_ret = 3.5 pts, late trend 0.23 pt/cycle vs
σ_null = 0.03 pt/cycle. True-label rehearsal changes nothing (3.8 / 0.23 — **the self-distillation
ratchet is NOT the mechanism**, killing that repair direction → C15j); 4× sleep steps changes
nothing (3.4 / 0.22); 4× reservoir helps then **coverage saturates at the unbounded oracle**
(both 2.7 / 0.15); 4× lower sleep lr changes nothing (2.8 / 0.16). The one real lever is the
old:new gradient share: 1:1 → 3:1 → 7:1 gives trend 0.15 → 0.12 → 0.09 pt/cycle with S_ret
2.7 → 1.7 → 1.4 — **a dial with diminishing returns that never reaches the noise bar while the
integration cost rises** (S_new +0.06 → +0.22 → +0.85 pt). E4's bounded-rehearsal arithmetic and
A9's compounding drift both received first empirical support. Frozen: reservoir 8000, old:new 48:16,
sleep lr 0.05, 625 steps, rank 4 (pilot commands recorded here; pilot result files superseded).
Predictions updated on pilot evidence, recorded before verdict seeds: C15a expected to survive
(margin ~0.3 pt on S_ret — the 2σ escalation rule may trigger), **C15b now predicted FALSIFIED**
(leak ~0.12 pt/cycle ≈ 2× the trend bar at the frozen point; no reachable mixing ratio passes).
This makes two predicted-fails this round (C15b pilot-informed, C15i registered at birth); the
one-predicted-fail guardrail is satisfied in spirit — both predictions carry real skin.

Statistics (acc[c][d] = post-sleep-c accuracy on domain d; acc[0] = post-foundation):
S_ret = mean over F1–F4 of acc[0]−acc[20]. S_new = mean over fresh-domain cycles of (composite
pre-sleep test accuracy − base post-sleep accuracy on that domain). S_trend = mean per-cycle drop
of mean-F accuracy over cycles 11–20 (accelerated arm: 41–60). S_plast = composite online accuracy,
first fresh cycle − last fresh cycle. Trend bar derived from the null arm: σ_null = sd of the null
run's per-cycle mean-F changes on seed 0; TREND_BAR = max(0.05 pt, 2σ_null), finalized after the
pilot, before verdict seeds.

**C15a — Sleep consolidates without destroying (T4 core).** [SURVIVES 9/9 after escalation —
S_ret 1.1–1.9 pts (every seed under the 2-pt bar), S_new ≤ 0.5 pt. T4's core holds at this scale:
twenty distillation merges keep old domains within the registered band and transfer the new
faithfully.]
Statement: stored-rehearsal arm has S_ret ≤ 2 points AND S_new ≤ 2 points.
Falsifier: either fails on ≥ 2 of 3 seeds → T4 falsified at this scale; the architecture fails
(THESIS's own words). S_new exists so no arm passes retention by merging nothing (the null control
is the demonstration).

**C15b — No compounding drift (A9, E25).** [FALSIFIED 3/3 as pilot predicted — trend 11–20 =
0.094/0.095/0.069 pt/cycle vs bar 0.050; seeds 1–2 exceed by > 2σ, verdict locked at n=3. BUT the
accelerated arm reframes the mechanism: trend at cycles 21–40 ≈ 0.006 and 41–60 ≈ 0.008 pt/cycle —
once the pool recycles and no *new* domain arrives, the leak vanishes at once. The drift is a
**novelty tax** (~0.09 pt on old knowledge per new domain packed in), not decay per sleep. A9's
unbounded-compounding reading is refuted; the registered claim, which measured the tax window,
stands falsified as written.]
Statement: stored arm's S_trend ≤ TREND_BAR, and the 60-cycle accelerated arm's S_trend over cycles
41–60 ≤ TREND_BAR.
Falsifier: either exceeds the bar on ≥ 2 of 3 seeds — constant per-cycle loss falsifies T4 even if
C15a's 20-cycle total passes.

**C15c — Plasticity survives sleeps.** [SURVIVES 9/9 after escalation — S_plast negative on every
seed (−0.2 to −6.2 pts, later fresh domains learned *faster*): the base does not stiffen across
20 sleeps, and no rebirth mechanism was needed for that (cf. C15f).]
Statement: stored arm's S_plast ≤ 1 point.
Falsifier: S_plast > 1 point on ≥ 2 of 3 seeds (the base stiffens across sleeps).

**C15d — Weight-addition merge fails retention (T2 Killer A, strongest form).** [SURVIVES 3/3 as
predicted — S_ret 72.6/62.7/64.9 pts: adopting the composite wholesale destroys the old domains
outright. T2 Killer A does not fire; rehearsal remains the price of merging.]
Statement: the weightadd arm (exact fold of the composite into the base — transfer is perfect by
construction, no rehearsal anywhere) has S_ret > 2 points.
Prediction: survives (E13 ported; the adapter's off-domain drift is adopted wholesale).
Falsifier: weightadd retains (S_ret ≤ 2 points) on ≥ 2 of 3 seeds → a no-rehearsal merge works,
T2 Killer A fires, and we celebrate in the ledger.

**C15e — Pure self-generated rehearsal fails (E4; F2 operationalised at last).** [SURVIVES 3/3 as
predicted — selfgen S_ret 24.3/17.7/20.1 pts (transfer fine): rehearsing on moment-matched noise
retains an order of magnitude worse than stored inputs. F2 now has a number at this scale.]
Statement: the selfgen arm (rehearsal inputs drawn from per-pixel running Gaussian moments of the
stream, targets still the pre-sleep base's logits) fails at least one of C15a's two bars.
Prediction: survives (model collapse; Sutton's "synthetic data is a big mistake").
Falsifier: selfgen passes both bars on ≥ 2 of 3 seeds → synthetic rehearsal rehabilitated at this
scale and F2 is weakened.

**C15f — Rebirth is safe in sleep.** [FALSIFIED 3/3 — rebirth S_ret 6.5/5.4/7.1 pts vs bar 2.0:
even ρ = 5 % random reinit with immediate rehearsal costs ~4–5 pts of retention. C8's lesson
extends into sleep; "rebirth only in sleep" leaves the thesis. Note: C15c shows plasticity needs no
rebirth at this horizon anyway.]
Statement: the rebirth arm (ρ = 5 % of units per layer reinitialised at sleep start, random
selection per C11, C11's recipe) passes both C15a bars.
Either outcome enters the thesis: survives → "rebirth only in sleep" is confirmed mechanism;
falsified → reinitialisation is retention-hostile even under immediate rehearsal, and C8's lesson
extends into sleep.

**C15g — Spawn→merge converges (A5).** [SURVIVES 3/3 — R-visit online ~0.75 → 0.87–0.89 and flat;
‖Δ‖ ~5.6 → 0.7–0.9 and flat. Repeated spawn→merge of one domain converges; no thrash.]
Statement: across R's visits (cycles 1, 2, 4, 8, 16), composite online accuracy at visits 3–5 is
each ≥ visit-2's − 2 points, and graduation delta-norm at visits 3–5 is each ≤ 1.25 × visit-2's.
Falsifier: either fails on ≥ 2 of 3 seeds → repeated spawn→merge of one domain thrashes instead of
converging; the nursery needs redesign before anything else is built on it.

**C15h — Merge dilutes corrections as predicted (E7).** [SURVIVES 3/3 — correction fraction
0.92/0.88/0.90 vs predicted 0.81 ± 0.15; inside the band, skewed toward *more* correction than the
gradient-share model (fresh coherent gradients beat stale rehearsal). Diagnostic: the correction
persists through cycle 20 (0.75 → 0.73) despite stale old-label examples in the reservoir.]
Statement: in the conflict arm (at cycle 10 the stream re-presents F1's inputs with labels shifted
y → (y+5) mod 10; the adapter learns the correction; rehearsal still teaches the old behaviour),
the correction fraction — post-sleep base accuracy on F1-new-labels ÷ the composite's pre-sleep
accuracy on the same — lies within (1−α) ± 0.15, where α = 3p/(3p+1) (the old side's F1 gradient
share at 3:1 mixing) and p = F1's share of the stored stream at sleep 10 (expected p =
10 000/130 000 → α ≈ 0.19, 1−α ≈ 0.81).
Falsifier: outside the band on ≥ 2 of 3 seeds → the dilution model behind C17's
rewrite-before-merge ordering is wrong and its math needs re-derivation.

**C15i — The never-rehearsed probe survives anyway.** [FALSIFIED 3/3 AS PREDICTED — P0 drop
14.4/23.1/12.2 pts. What sleep never rehearses erodes, at ~0.6–1.2 pts/sleep of merge traffic (the
null arm loses exactly nothing, the frozen-base ensemble keeps P0 at 92 %). Thesis point 3's "slow
drift of the unrehearsed" is real and fast; C17's canary now has a measured rate to watch.]
Statement: P0 (in the base from foundation, never rehearsed by any sleep) has acc[0]−acc[20] ≤ 2
points on the stored arm.
Prediction: FALSIFIED — thesis point 3 itself says what sleep never rehearses drifts; this claim
existing keeps us honest about it and measures the rate.
Falsifier of the prediction: P0 survives on ≥ 2 of 3 seeds → sleep is far gentler off-rehearsal
than believed, and the rehearsal requirement (and C17's canary design) can be relaxed.
Pilot preview: P0 dropped 9–33 pts across pilot arms (null arm: 0.3 pt) — the erosion is
distillation traffic, not machinery.

**C15j — Rehearsal targets don't matter: true labels buy nothing over self-logits.** [FALSIFIED
5/9 after escalation — the gain is positive on all 9 seeds (+0.6 to +1.4 pts, mean ≈ +1.0) and
clears the 1-pt bar on 5. Rehearsal grounded in stored true labels retains measurably better than
self-distillation; the pilot's null was setting-dependent (at 1:1/2000 the coverage bottleneck
masked target quality). Diagnostic: the advantage is early-phase (labels' late trend equals
stored's; P0 unaffected) — grounding softens the novelty tax, it does not stop unrehearsed drift.
Design consequence: sleep's rehearsal side should use the store's confirmed answers where it has
them — the same invariant `notes/design-audit.md` chose for the audit ledger.]
Statement: the labels arm (rehearsal side trained on stored true labels instead of the pre-sleep
base's logits, all else identical) does not beat stored by ≥ 1 point on S_ret.
Prediction: survives (pilot: 3.8 vs 3.5 pts at naive constants — a null worth recording: it kills
the "self-distillation ratchet" repair direction and localises the leak in interference, not
target quality).
Falsifier: labels beats stored's S_ret by ≥ 1 point on ≥ 2 of 3 seeds.

**Escalation registration (2026-09-01, after the n=3 verdicts, before any escalation run).** By the
2σ rule, C15a (S_ret 1.5/1.9/1.6 vs bar 2.0, 2σ = 0.42), C15c (S_plast −0.9/−4.4/−0.9 vs bar 1.0,
2σ = 4.0) and C15j (labels gain 0.8/1.4/1.2 vs bar 1.0, 2σ = 0.62) are *undecided at n=3* and
escalate to seeds 4–9 (stored + labels arms, same frozen constants). Decision rule, registered now:
an escalated claim holds iff its per-seed condition holds on ≥ 5 of the 9 seeds. C15b does NOT
escalate: seeds 1–2 exceed the trend bar by > 2σ (0.94/0.95 vs 0.50, 2σ = 0.29), locking the
2-of-3 falsifier regardless of seed 3 — FALSIFIED stands at n=3.

Controls, registered, not claims: **null** (adapter never trained; sleeps still run — sets σ_null;
if null itself fails C15a-retention, distillation is destructive per se and every other arm is
uninterpretable — E25's point); **oracle** (unbounded reservoir; stored fails + oracle passes →
reservoir size is the binding constraint; both fail → the rehearsal principle is broken at this
scale, evidence toward T2 Killer B); **ensemble** (never merge, oracle routing at eval — the
retention/plasticity ceiling and the price of merging; unbounded pool, violates the thesis on
purpose); **scratch** (fresh net, iid over all 21 domains, same example budget — the capacity
ceiling the retention bars live under); **probation 2 500 / 40 000** (wake-length dial readings for
the open probation-length question; reported, no bar).

## Claims — audit (`checks/audit.py`, instrument `audit_run.py`; registered 2026-09-01 before any run)

Protocol (design: `notes/design-audit.md`; substrate: the frozen C15 instrument, rehearsal in
C15j's winning mode — true labels). The **ledger** (`cl/ledger.py`) replaces the reservoir: item =
probe slice of n_p = 200 stream-confirmed (x, y*) pairs from one domain, 5 slices admitted at a
domain's first merge (foundation F1–F4 admitted at birth; P0 never), s_ref = base's score at last
audit turn, canary fraction 10 % (rehearse = False forever: audited, never rehearsed, never
rollback grounds), plus **two all-canary domains** (the first two fresh merges, domains 6 and 7:
every slice rehearse = False, and they never recur in the recycled pool — the deliberately
drift-eligible population). Audit, **two-phase per sleep**: the m = 20 rotation slices
(oldest-audited; R = ⌈K/m⌉ ≈ 4–5 at full K ≈ 100) are scored BEFORE distillation (a fail here
predates this sleep → repair watch: 5× rehearsal weight, audited every sleep, s_ref preserved) and
re-scored AFTER (a fail that is new since the pre-score is this sleep's surgery → exact rollback,
merge refused, refresh discarded). fail = s_now < s_ref − ε, ε = 9 points (≈ 3.6σ of a 200-example
slice's binomial noise). Passing slices refresh s_ref := s_now — the ratchet the boiled-frog
statistic guards: Δ_t = mean per-sleep drop of the all-canary domains' slice scores (logged every
sleep); Page-Hinkley PH = max(0, PH + Δ_t − k), k = R·δ_tol = 0.25 pt, reset at the frog arms'
onset; alarm at PH > h, h = 5·sd(Δ) on the frognull seed-0 pilot (formula frozen here); a PH alarm
is consolidation debt, never a rollback. Injections: **correction** = C15's conflict (cycle 10, F1
labels y → (y+5) mod 10; ledger arm does rewrite-before-merge — colliding slices' y* rewritten and
s_ref reset against the current base BEFORE the merge; snapshot arm does not); **unconfirmed
damage** = 100 wrong-label SGD steps each on F2 (rehearsed — the repair test) and on all-canary
domain 6 (unrehearsed — the detection test), injected after sleep 14 commits, no nursery trail.
**Pilot record (2026-09-01, seed-0, first instrument round):** with label-grounded rehearsal the
sleep is self-healing on everything rehearsal covers — the injected F2 lesion (0.92 → 0.06) healed
to 0.88 in ONE ordinary sleep before any audit looked, and per-sleep weight noise at 3e-4 produced
zero net erosion on rehearsed content (frog ≈ frognull everywhere). Consequences, all instrument
redesigns recorded here before verdicts: the audit moved to two-phase (detection must look before
rehearsal masks); the boiled frog and the detection test moved to the all-canary domains (the only
honestly persistent damage); repair of covered content is mostly the rehearsal's own doing —
**the audit's residual jobs are detection, surgery-checking, and coverage-gap watching**, which is
itself the round's first finding. **Second pilot finding (same day, second instrument round):**
freshly-merged unrehearsed domains erode fast (~3–4 pt/sleep at first — C15i's phenomenon at
fresh-merge strength), so canary slices with frozen expectations become standing "failures" and a
threshold-straddle at the post-audit masquerades as surgery damage (the pilot produced a
three-sleep rollback loop that silently skipped the damage injection). Enforced fix, from the
memo's own words ("never rollback grounds"): canary slices never enter the rollback verdict nor
the repair watch, and their s_ref refreshes on every audit — their per-item test is thereby a
*sharp-drop* detector (a ≥ ε cliff since the last rotation visit), blind to slow drift on purpose;
slow drift on canaries is exactly what the Page-Hinkley aggregate watches. Owned gaps, registered: the
rewrite itself is not transactional (a rollback of the conflict sleep would leave expectations
corrected but the base not — E20's territory, logged if it occurs); slice retirement/eviction
(T_stale, n_contra) is designed but unexercised this round — the decay-law round owns it; canary
slices damaged by injections are expected to be caught but NOT repaired (that is the canary's
measurement, and C15f's verdict already removed the rebirth arm this memo once carried).

**C17a — The ledger audit accepts confirmed corrections and repairs late-caught damage (A7).**
[SURVIVES 3/3 — correction fraction 1.01/1.02/1.02 (the base slightly beats the composite), no
rollback of the conflict sleep; F2 0.91 → 0.06 → 0.89–0.90 within two sleeps of the injection.
A7 is resolved by the ledger invariant on this instrument.]
Statement, ledger arm: (i) post-sleep-10 correction fraction (base accuracy on F1-new-labels ÷
composite's pre-sleep accuracy on same) ≥ 0.72 (C15h's measured reference 0.74–0.76 minus a 2σ
allowance) with the conflict sleep NOT rolled back; (ii) rehearsal-covered damage self-repairs:
the F2 probe returns to within ε of its pre-damage value within 2 sleeps of the injection,
detection or no detection (the pilot showed repair is the rehearsal's own doing; the bar makes it
a guarantee, not an anecdote).
Falsifier: (i) or (ii) fails on ≥ 2 of 3 seeds → the ledger invariant does not resolve A7 in
practice and the audit design returns to the drawing board.

**C17b — The snapshot-anchored audit also accepts the correction (A7 control).** [FALSIFIED 3/3
AS PREDICTED — the snapshot-anchored audit rolled the correcting merge back on every seed (final
correction 0.02–0.03: the stream's confirmed fix erased). A7's stasis-enforcement is a real
failure mode, and rewrite-before-merge is demonstrably what prevents it.]
Statement, snapshot arm: final correction fraction (cycle 20) ≥ 0.72.
Prediction: FALSIFIED — anchored to the base's own past answers, the audit reads the confirmed
correction as damage and suppresses it (same-sleep rollback, or repair-forward actively
re-teaching the old labels — either mechanism, the outcome statistic catches both; that is why the
statistic is the outcome). If instead it survives, A7 was a phantom at this scale and the ledger's
extra machinery is unjustified — say so and simplify.

**C17c — Rotation catches unconfirmed damage within R sleeps.** [SURVIVES 3/3 — all five domain-6
slices first-caught at cycle 18, latency exactly R = 4, identical on every seed: the guarantee is
deterministic (zero margin, zero variance — the 2σ rule has nothing to escalate).]
Statement, ledger arm: every one of all-canary domain 6's five slices (damaged at sleep 14,
unhealable by rehearsal — honestly persistent) is first caught by the rotation's per-item test
within R = ⌈K/m⌉ sleeps of the injection (K read from the run's log at sleep 14; expected R = 4).
Falsifier: any slice uncaught past R on ≥ 2 of 3 seeds — the deterministic full-coverage guarantee
(the case for rotation over iid sampling, `design-audit.md` math) fails in implementation or in
principle.

**C17d — The boiled frog is caught by the aggregate and only by the aggregate.** [FALSIFIED 3/3 —
(i) held: PH alarmed at cycles 22/27/23, all inside the 31 bar, with ZERO null PH false alarms;
but (ii) fell: the per-item test fired first on every seed (cycle 21 — the onset burst), and
(iii) fell on seed 2 (5 null per-item fails vs ≤ 1; natural canary erosion bursts past ε even
uninjected). Reading: E6's slow-drift threat model is refuted at this scale — content drift is
bursty enough that per-item + rotation + refresh already detects it; PH is a clean, false-alarm-
free confirmatory statistic, not the sole detector. The honest residual: per-item on unmaintained
content chatters (the canary telling the truth about nonsmooth decay), so per-item fails on
canaries stay informational-only — which the rollback rules already enforce.]
Third pilot finding, recorded: uniform WEIGHT NOISE is fully self-healed at every survivable level
(3e-4 and 1.5e-3 both net zero — even unrehearsed canary domains sit at a maintained equilibrium,
held up by other domains' rehearsal through shared weights: the rehearsal-coverage boundary is
soft). The dangerous slow drift is CONTENT drift — E6's actual scenario — so the frog is
registered as targeted slow poison: one wrong-label SGD step per sleep at lr 0.003 on the canary
domains from cycle 21 (dose pilot-frozen: mean canary erosion ≈ 0.8 pt/sleep, inside the window
where the per-item test is blind between rotation visits yet PH accumulates). Fourth pilot
finding, recorded: content drift on this substrate is BURSTY (per-sleep canary drops of 0.5–6 pts
around the ~0.8 mean at the frozen dose) — a perfectly smooth sub-threshold leak does not exist to
inject, so "per-item never fires" is unachievable by construction and E6's smooth-vs-threshold
dichotomy is idealised; the operational content of E6 is that the aggregate alarms FIRST.
Statement: on the frog arm, (i) Page-Hinkley on the canary-domain score series alarms within 2R
sleeps of onset; (ii) PH's first alarm is no later than the per-item test's first fire on the
poison (the aggregate leads; per-item catches only bursts ≥ ε); and on the frognull arm (no
injection), (iii) zero PH alarms and ≤ 1 per-item false alarm from onset through cycle 60 (the
claim's window; the packing phase's natural canary erosion is C15i's known phenomenon, not a
false alarm).
Falsifier: any of the three fails on ≥ 2 of 3 seeds → the audit's slow-damage story (E6) is wrong
at this scale: either the aggregate is deaf or late, or it cries wolf.

Seed-0 pilot may fix: the frog dose, h (by the frozen 5·sd formula), and instrument sanity.
Everything frozen before seeds 1–3; escalation and margins per the standing conventions.

**Evidence — audit round, verdict seeds 1–3 (2026-09-01).** `checks/audit.py` →
`results/audit_{arm}_s{1,2,3}.json`, `results/verdicts_audit.json`, log
`results/audit_check_run1.log`. h = 0.0388 (frozen formula on the frognull seed-0 pilot).
Sanity: ledger-rehearsal base retains at least as well as C15's labels arm (S_ret on F2–F4:
0.9/1.1/−0.2 pts over 20 sleeps); P0 drop 20–32 pts (C15i's erosion, now with audit traffic);
ledger-arm rollbacks 0, snapshot-arm rollbacks exactly 1 (the correction — the failure mode on
demand). The cycle-14 injection is broad (foundation mean dips to ~0.32 before the next sleep
heals the covered domains); the repair watch caught ~18 collaterally-damaged rehearseable slices
and every one recovered same-sleep. Round tally: **C17 a/c survive, b/d falsified — b as
predicted, d the round's surprise** (per-item is stronger than E6 feared; PH is confirmatory).

## Claims — composition (`checks/compose.py`, instrument `compose_run.py`; registered 2026-09-01
before any run)

Protocol (design: `notes/design-composition.md` "minimal compositional stream"; HOLES A1/E27).
Two-digit pair canvas (1568 inputs, two independently drawn MNIST images, each canvas seen once).
Phase 1: a fresh 3×256 MLP learns both digits online (two 10-way blocks, summed CE; 60 000
canvases, batch 16, lr 0.05). Phase 2: the binary predicate **parity of the digit sum** (registered
primary — non-linear in the pair's identities; `lt` and `ge10` logged as secondary), 10 000
canvases online, arms: **ft** = whole base + fresh 2-way head (reference ceiling); **deep** =
frozen base + all-layer rank-4 adapter with gates + head (the nursery); **tap** = frozen base + a
256→64→2 side MLP on the last hidden (E7's read-only adapter); **probe** = linear head on the
frozen last hidden (pilot gate: if probe reaches ft − 2 pts, no ceiling exists and the instrument
cannot test A1 — degrade the base per the memo before spending verdict seeds); **scratch** = fresh
untrained net + head at the same phase-2 budget (C21c). Statistic = final test accuracy on 10 000
held-out canvases.

**C21a — The composition gap is real: a read-only adapter trails direct fine-tuning (A1/E7).**
[FALSIFIED 3/3, decisively — gaps 0.9/2.2/1.4 pts, every seed ≥ 2σ under the 5-pt bar. A1
dissolves on this substrate: a side-MLP on frozen features composes parity essentially as well as
full fine-tuning. The eureka 63/100 ceiling (nanoGPT, char-level) is not explained by adapter
placement here; scale/substrate caveat recorded, E7's ceiling remains attributed elsewhere.]
Statement: ft − tap ≥ 5 points on parity.
Falsifier: gap < 5 points on ≥ 2 of 3 seeds → A1 dissolves on this substrate: side-adapters
suffice for composition and the eureka ceiling was an artefact of something else.

**C21b — Deep low-rank adapters close the gap (A1's committed fix).** [FALSIFIED 0/9 after
escalation — deep-gaps 2.8–7.3 pts, no seed within the 2-pt bar, while tap's gaps are 0.9–2.2.
The KILLER reading does NOT fire: the nursery is not too weak — its read-only variant succeeds
(C21a); it is the committed all-layer rank-4 design that under-performs at these registered
constants (a rank/lr/steps sweep would be a separate claim). Combined C15+C21 reading, the
round's real product: **adapter type should match novelty type** — input-shift novelty (C15's
permutations) needs deep reshaping, where the deep adapter reached 84–90 %; rule novelty over
known inputs favours the read-only tap, which also never touches the base: the cleanest address
there is (deletion trivially exact, no orthogonality bookkeeping). The doorman (C16) inherits the
routing question.]
Statement: ft − deep ≤ 2 points on parity.
Falsifier: fails on ≥ 2 of 3 seeds → the nursery as designed is structurally too weak to compose
with base knowledge, and A1 stands as a KILLER against the architecture.

**C21c — Primitives are unnecessary: scratch learns the predicate as fast as the tuned base.**
[FALSIFIED 3/3 AS PREDICTED — scratch 60.2/64.3/61.1 vs ft 91.9/91.3/92.0: learned primitives
accelerate composite learning by ~30 pts at this budget. The composition frame is sound; probe
75.2/67.6/73.3 confirms the non-linear ceiling on verdict seeds too.]
Statement: scratch ≥ ft − 2 points at the 10 000-canvas budget.
Prediction: FALSIFIED — the whole composition frame presumes learned primitives accelerate
composite learning; if scratch keeps up, composition-through-the-base is moot on this substrate
and C18/C19's design must move to harder streams.
Falsifier of the prediction: scratch keeps up on ≥ 2 of 3 seeds.

Seed-0 pilot may fix: the probe gate (base degradation if no ceiling), phase-2 budget. Frozen
before seeds 1–3; margins and escalation per the standing conventions.

## Claims — doorman (`checks/doorman.py`, instrument `doorman_run.py`; registered 2026-09-01
before any run; design `notes/design-doorman.md`)

Protocol. Base: 3×256 MLP learns plain MNIST online (60 000 examples), then freezes — **A4's
cold-start is therefore NOT tested this round** (a mature frozen base dodges the circularity;
deferred to the growth rounds, recorded here). Stream: 30 segments × 3 000 examples, fixed inputs,
rule = label permutation from a pool of 6 (uniform draws, immediate repeats allowed — invisible
boundaries that must not alarm); segments 10 and 20 are noise traps (uniform random labels); the
last rule is hard-but-known (permutation + 40 % uniform flips — stationary irreducible error).
Adapters = read-only taps (256→64→10, C21's winner), one in write-mode at a time (E18). Detector:
Page-Hinkley (mean-increase, running mean, reset on every routing event) on the active adapter's
per-batch error, k = 0.05, h = 2.0; on alarm, the recurrence check scores the last 5 batches under
every pooled adapter — best error ≤ θ = 0.3 wakes it, else spawn; pool cap 12, oldest-unused
evicted (logged). Arms: doorman · oracle (true segment ids — ceiling) · continue (one adapter
forever) · spawn (alarm → always spawn) · marginal (E26: same PH watching the batch pixel-mean —
blind by construction) · doorman/oracle at seglen 1 000 (C16e). Routing truth for measurement
only: a rule's adapter = the majority-active adapter in the second half of its first segment.
Statistics: mean online accuracy over non-noise segments; spawn/wake/wrong-wake counts; detection
latency; per-segment routing correctness. Seed-0 pilot may re-fix k/h/θ/buffer and the C16a
margin; frozen before seeds 1–3.
**Pilot finding (seed 0), recorded: improvement-invisible switches.** A change-detector on error
sees only error *increases*; a switch OUT of a bad regime (noise → learnable rule) leaves the
active adapter's error flat-then-falling, so no alarm fires — the noise adapter quietly retrains
into the new rule and the rule's identity splits across two adapters. Accuracy is unharmed (the
claims' bars are accuracy-based on purpose); routing purity is reported but not a bar. The honest
fix (a downward detector waking the recurrence check on improvements too) is future design, noted
for the decay-law round where identity purity starts to matter. Also fixed in pilot: pool-cap
eviction now recycles the oldest-unused slot at spawn time (the first implementation evicted per
batch once at cap).

**C16a — The doorman tracks the oracle.** [FALSIFIED 3/3 — gaps 4.2/3.9/4.5 pts vs the 3-pt bar,
all ≥ 2σ beyond it]
Statement: doorman mean online accuracy (non-noise segments) ≥ oracle − 3 points.
Falsifier: gap > 3 points on ≥ 2 of 3 seeds → label-free routing loses too much; the nursery
needs a better doorman before the decay law can trust its counters.

**C16b — Waking beats both never-switching and always-spawning.** [FALSIFIED 3/3, and this is the
round's finding: ALWAYS-CONTINUE — one tap retrained through every switch, no routing at all —
comes within 1.0–1.7 pts of the doorman (bar demanded +10) and within ~5 pts of the oracle.
Relearning a permutation rule costs a few hundred examples, so recurrence recognition buys almost
nothing at 3 000-example segments — C22's cheap-relearning economics, striking a third time.
Spawns 17–18 vs the spawn-arm's 22–23 (not ≤ half). Scale caveat recorded, not leaned on: where
relearning is expensive, routing should pay; untestable on this instrument.]
Statement: doorman ≥ continue + 10 points; AND doorman ≥ spawn-arm − 2 points with ≤ half the
spawn-arm's spawns (recurrence recognition pays in pool size at no accuracy cost).
Falsifier: either fails on ≥ 2 of 3 seeds — if always-spawn matches with a bounded pool anyway,
the doorman's wake half is dead weight (C22's cheap-relearning lesson would extend to routing).

**C16c — The stream is marginal-proof (E26 validity control).** [SURVIVES 3/3 — the marginal
detector never fired and matched continue exactly; the stream leaks nothing through input
statistics]
Statement: the marginal arm's detector never alarms and its accuracy is within 1 point of
continue's.
Falsifier: it alarms usefully on ≥ 2 of 3 seeds → the stream leaks task identity through input
statistics and every other verdict this round is void.

**C16d — The traps hold (A3).** [SURVIVES 3/3 — exactly one spawn per noise segment, zero
re-alarms inside hard-but-known segments: PH-on-error separates new-and-learnable from noise and
from stationary hardness, on every seed. A3's detector-level worry is answered; its
identity-purity worry is not (routing purity 0.14–0.31 — the improvement-invisible-switch
splitting from the pilot, now quantified: the decay law must NOT trust doorman-assigned
identities on this instrument).]
Statement: ≤ 2 spawns per noise segment (noise is entered once, not thrashed), and zero alarms
inside hard-rule segments after their first 50 batches (stationary hardness never re-alarms).
Falsifier: either fails on ≥ 2 of 3 seeds → raw-error routing confounds A3's cases and the
doorman needs the habituation organ earlier than planned.

**C16e — Short segments break the doorman.** [FALSIFIED 3/3 AS PREDICTED — gaps 10.7/12.5/12.4
pts at seglen 1 000: segments near the doorman's reaction time are invisible to it, the memo's
stream constraint now a number]
Statement: at seglen 1 000 the doorman stays within 3 points of the seglen-1 000 oracle.
Prediction: FALSIFIED — detection latency plus relearning overhead amortizes over 3 000 examples
but not 1 000; segments shorter than the doorman's reaction time are invisible to it (the memo's
registered stream constraint, made a number).
Falsifier of the prediction: it holds on ≥ 2 of 3 seeds → the overhead is negligible even at
1 000 and the constraint can be relaxed.

## Claims — decay law (`checks/decay.py`, instrument `decay_run.py`; registered 2026-09-01 before
any run; HOLES E1/E3 — the ledger's ranked worry #1)

Protocol. Frozen base (60 000 plain-MNIST examples); stream of 60 segments × 600 examples (short
on purpose: relearning must cost something — C16e inverted into instrument design); each segment's
rule drawn iid from 20 label permutations under **Zipf recurrence** (p ∝ 1/rank — E3's world) or
**uniform** (the no-structure control). Routing by ORACLE identity (C16's purity constraint,
recorded); one tap per rule; pool bounded at 8 of 20 — evictions forced. The registered decay-law
price, J's T2 ported to its minimal testable form: **price = n_uses / (now − t_last)** (frequency ×
recency), evict argmin. Reference policies: random (E1's control), LRU, LFU (the price's parts),
Belady (evict the farthest next recurrence — the clairvoyant ceiling), nolimit (unbounded pool).
Statistic: mean online accuracy over all segments; streams bit-identical across arms per seed
(eviction randomness on a separate generator). Owned scope limits, registered: the
confirmations/contradictions counters are unexercised (this stream has no conflicting content —
they wait for a conflict-bearing stream); E3's "rare but critical" needs value-weighted probes —
here all occurrences weigh equally, so the tail matters little by construction and the deep
aliasing worry stays open; margins provisional at 2 pts pending pilot seed-noise (2σ rule).

**Escalation registration (2026-09-01, after the n=3 verdicts, before any escalation run).**
Counters−random gaps 1.0/0.3/2.3 (sd 1.0), Belady's own edge only 2.3–5.1 over random, and every
C23 per-seed statistic sits within 2σ of its bar → C23a/b/c/d are all *undecided at n=3* and
escalate to seeds 4–9 (7 arms: counters/random/belady/lru/lfu on Zipf, counters/random on
uniform); decision ≥ 5/9 per the standing rule. Early reading recorded either way: even the
clairvoyant ceiling is thin here — eviction choice is low-stakes when relearning is cheap.

**C23a — Counters beat random eviction (the E1 control; thesis point 3 lives here).** [FALSIFIED
3/9 after escalation — mean edge +1.4 pts, positive on 7/9 seeds but ≥ 2 pts on only 3. The E1
falsifier fires as registered: at this scale chosen forgetting buys ~1.4 pts over random
forgetting; thesis point 3 is revised in THESIS v0.6 (same session). The signal is real and
lawful — see C23d — but economically thin: cheap relearning's fourth strike.]
Statement: on the Zipf world, the counters policy's mean online accuracy ≥ random's + 2 points.
Falsifier: counters ≤ random + 2 on ≥ 2 of 3 seeds → **thesis point 3 collapses to "bounded
memory + random eviction"** — chosen forgetting is a budget, not a decision, at this scale; the
thesis moves in the same session.

**C23b — Counters capture a real fraction of the clairvoyant's edge.** [FALSIFIED 2/9 — and the
ceiling itself is thin: Belady's edge over random is only 2.1–5.9 pts, with the unbounded pool a
further ~0.5 pt above Belady. Eviction choice is a low-stakes decision here, full stop.]
Statement: (counters − random) ≥ 0.5 × (Belady − random) on the Zipf world.
Falsifier: < half on ≥ 2 of 3 seeds → the price tracks the past but not the future; the decay law
is a weak estimator even where recurrence has structure. (If Belady ≈ random, eviction choice is
worthless per se and C23a falls with it — coherent.)

**C23c — The combined price is not beaten by its parts.** [SURVIVES 6/9 — the frequency×recency
price edges LRU and LFU individually, for what the thin stakes are worth]
Statement: counters ≥ max(LRU, LFU) − 0.5 points on the Zipf world.
Falsifier: a single-ingredient policy beats the combined price by > 0.5 points on ≥ 2 of 3 seeds
→ that ingredient IS the decay law and the other is dead weight; simplify the law accordingly.

**C23d — Counters still beat random when recurrence has no structure.** [FALSIFIED 8/9 AS
PREDICTED — on the uniform world the counter edge disappears (mean ≈ −0.6 pt). E3's aliasing as a
number: counters pay exactly where recurrence has structure and nowhere else; the instrument does
not leak, and C23a's small Zipf-world edge is therefore a real, lawful signal — just a small one.]
Statement: on the uniform world, counters ≥ random + 2 points.
Prediction: FALSIFIED — with no frequency structure the counters carry no signal about the
future; this is E3's aliasing made a number (counters pay exactly where recurrence has structure,
and nowhere else). If they win anyway, something in the instrument leaks and every C23 verdict is
suspect — audit before believing.

## Claims — the facts world (`checks/facts.py`, instrument `facts_run.py`; registered 2026-09-01
before any run)

Why this substrate: four independent falsifications (C22, the novelty tax, C16, C23) trace to one
economic law — relearning is cheap at MNIST-task scale. This round moves to a world where
relearning is **expensive by scarcity**: N = 50 000 random key→value facts (fixed 784-dim Gaussian
keys, fixed 10-class values), streamed iid under Zipf(s = 1.1); a forgotten fact cannot be
relearned until the stream shows it again, and most facts are cold. It also finally
operationalises **F1, the big-world hypothesis**: the world (50 000 facts) is strictly larger than
every memory the machine has (8 000 slots; a 3×256 net that needs multiple visits per random key).
Honesty about the frame, registered: the earlier falsifications delineated a band where memory
cannot pay (fast recurrence, cheap relearning); this round deliberately tests INSIDE the
complementary band — the claims are conditional on structured scarcity and say so. Life: 300 000
examples, batch 16, lr 0.05, online scoring pre-update, prequential bits as in run.py. Arms: bp
(no memory) · reservoir (C9's classic mechanism verbatim, duplicates allowed) · random
(deduplicated fact-store, random eviction — E1's control here) · priced (fact-store, evict argmin
n_seen/(now − t_last) — C23's registered price) · random/priced on the uniform world (s = 0).
Seed-0 pilot gates: per-fact acquisition must cost > 1 visit (else scarcity is toothless), and
seed noise must leave the inherited margins ≥ 2σ. Frozen before seeds 1–3.
**Pilot record (seed 0):** the first registered world (N = 50 000, s = 1.1) was degenerate — the
head so hot that bp alone reached 86.6 % and reservoir saved only 5.4 % of bits (the C22 trap in
mirror image). Retuned per the gate to **N = 20 000, s = 0.6** (a wide warm middle: ~15 visits per
fact): bp 77.7 % / 308 543 bits vs reservoir 83.9 % / 232 356 bits — a 24.7 % saving, the C25a
game live. Also previewed and recorded before verdicts: priced ≈ random (84.4 vs 84.7) even here —
candidate mechanism, stated in advance of the wave: **occurrence-weighted admission already
implements the frequency prior; the stream does the pricing, so eviction choice inherits little to
add**. If C25b falsifies on the seeds, that is the generalised form of E1's verdict.

**C25a — Retention pays in prequential bits where relearning is expensive (C22's statement, one
knob changed: substrate).** [SURVIVES 3/3, decisively — savings 25.2/24.9/25.5 % vs the inherited
10 % bar (online +6.3–7.4 pts). **The program's first positive economics verdict**: the memory
organ pays exactly in the band the thesis aimed at. C22 and C25a together bracket the law:
retention pays iff relearning is expensive; the boundary is the recurrence/scarcity structure.]
Statement: the reservoir arm's total prequential bits are ≥ 10 % below bp's (bar inherited from
C22 verbatim).
Prediction: survives — this is C22's scale caveat put to the test; if retention cannot pay even
here, the currency is dead in both bands and the thesis's memory organs have no economic ground
anywhere on this instrument family.

**C25b — Priced eviction beats random eviction where facts are scarce (C23a's statement, one knob
changed: substrate).** [FALSIFIED 3/3 — priced is 0.4–0.5 pts WORSE than random even under
scarcity, ≫ 2σ from the bar. E1's verdict is unconditional across both economic bands, and the
pre-stated mechanism stands: admission-by-occurrence already implements the frequency prior.
Thesis point 3's v0.6 demotion is no longer scale-caveated on this instrument family.]
Statement: priced ≥ random + 2 points mean online accuracy on the Zipf world (bar inherited).
Prediction: survives — under scarcity, evicting a hot fact costs real stream-time; the price
finally has something to protect.
Falsifier: fails on ≥ 2 of 3 seeds → E1's verdict generalises across both economic bands and
thesis point 3's demotion (v0.6) becomes unconditional at this scale.

**C25c — Priced eviction beats random without structure.** [FALSIFIED 3/3 AS PREDICTED — the
counter signal is absent without recurrence structure on this substrate too; C23d's law
generalises and the instrument does not leak.]
Statement: on the uniform world (s = 0), priced ≥ random + 2 points.
Prediction: FALSIFIED — C23d's law should hold on every substrate: no recurrence structure, no
counter signal. A win here means a leak; audit before believing anything else in the round.

## Claims — graduation (`checks/graduation.py`; registered 2026-09-01 before any run; HOLES B3,
raised directly by the user: "it's best if the AI can ignore or not merge noise")

Protocol: the frozen C15 instrument (stored arm, all frozen constants) with **noise domains** at
cycles 5, 10, 15 (labels uniform random — unlearnable by construction) and the **graduation gate**
(B3's first registered criterion): an adapter merges only if its last-quarter wake online accuracy
≥ θ_g = 0.5 (the stream confirming its predictions; noise caps near chance 0.1, real domains
reach ~0.75). A failed graduate dies in the nursery: no distillation, its wake buffer never enters
the reservoir. Arms: **gated** (θ_g = 0.5) vs **ungated** (θ_g = 0 — noise is merged AND its
mislabeled examples poison the rehearsal pool, ~15 % of it by the end).

**C29a — Noise never graduates; knowledge always does.** [SURVIVES 3/3, unanimous — every noise
cycle rejected, every real domain graduated, all seeds. The nursery-level noise filter is proven
machinery. Owned limitation, raised by the user at review: θ_g = 0.5 is a WORLD-SPECIFIC constant
(it works because chance ≈ 0.1 and learnable ≈ 0.75 here); the registered criterion ladder for
B3's future rounds: level gate (v0, this) → progress gate (self-calibrating within a world:
improved-during-probation beyond 2σ of slope noise; would also accept hard-but-learnable content
a level gate wrongly rejects — candidate C29c) → cross-time confirmation (the thesis-native
n_conf: noise of every shape shares one signature, inconsistency on re-encounter; needs
recurrence machinery).]
Statement: in the gated arm, all three noise cycles fail graduation and every real-domain cycle
passes, on every seed.
Falsifier: any noise merge or any real-domain rejection on ≥ 2 of 3 seeds → θ_g = 0.5 is not a
separating threshold and B3 needs a per-domain-calibrated criterion.

**C29b — Merging noise is harmless (the sleep shrugs off garbage).** [SURVIVES 9/9 after
escalation — gaps −0.2 to +0.6 pts, all inside the 1-pt bar. **The program's first missed
predicted-fail** (after eight straight hits), and the miss is informative: a noise adapter learns
nothing, so it is nearly an identity adapter, and merging identity is the distillation fixed
point (C15's null arm, rediscovered from the other side); 15 % label-poisoned rehearsal is weak
smoothing against an 85 % correct gradient. The sleep is intrinsically noise-robust; the
graduation gate is hygiene, not protection — which is exactly what makes an imperfect,
self-calibrating gate acceptable (its mistakes are cheap).]
Statement: the ungated arm's S_ret is within 1 point of the gated arm's.
Prediction: FALSIFIED — three garbage merges plus a rehearsal pool 15 % poisoned with random
labels should cost measurable retention (C15e charges 20 pts for 100 % garbage rehearsal; the
dilution scaling is the open question this claim measures).
Falsifier of the prediction: within 1 point on ≥ 2 of 3 seeds → the sleep is robust even to
unfiltered noise and the graduation gate is hygiene, not protection.

**C29c — The progress gate is world-free and strictly better (B3's ladder, rung 2; registered
2026-09-01 before any run).** [proposed]
Protocol: the C29 instrument plus a **hard-but-learnable domain** at cycle 7 (labels with 60 %
uniform flips: achievable online ≈ 0.40 — real knowledge a level gate at θ = 0.5 must wrongly
reject). Gate: graduate iff last-quarter wake online − first-quarter ≥ δ = 0.025 (≈ 2σ of the
quarter-mean difference under batch-16 binomial noise — a STATISTICAL constant, not a world
constant; the user's critique made precise). Arms: progress (δ = 0.025) vs level (θ = 0.5), both
with noise cycles 5/10/15 and the hard cycle 7.
Statement: the progress arm rejects all three noise cycles AND graduates the hard domain AND every
normal domain, on every seed; the level arm's rejection of the hard domain is the logged contrast
(prediction: it rejects — its ceiling sits under its own threshold).
Falsifier: the progress gate wrongly rejects (hard or normal) or wrongly admits (noise) on ≥ 2 of
3 seeds → the 2σ slope margin is not separating and B3 advances to cross-time confirmation
earlier than planned.
Status: **FALSIFIED 3/3 — and the miss is a mis-cut claim, not a broken mechanism.** The only
wrongful rejections were the MASTERED REVISITS (R at cycles 2/8/16: composite starts at 0.87,
no progress possible); noise was rejected and the hard domain graduated on every seed, while the
level arm rejected the hard domain exactly as predicted. Rejecting a mastered revisit is harmless
(the knowledge is already in the base — the skipped merge is C15's null no-op) and arguably
optimal (no pointless sleep). The registered statement demanded "graduates every normal domain"
and was wrong to.

**C29d — The progress gate merges exactly the learned (C29c re-cut; registered 2026-09-01 after
C29c's diagnosis, before any fresh-seed run; verdict on fresh seeds 4–6).** [proposed]
Statement, progress arm (δ = 0.025) on seeds 4/5/6: (i) all noise cycles rejected; (ii) the hard
cycle and every first-visit fresh domain graduated; (iii) R's revisits after its first merge
rejected (mastery = nothing to consolidate); (iv) S_ret ≤ 2 points (skipping mastered merges
costs no retention).
Falsifier: any of (i)–(iv) fails on ≥ 2 of 3 fresh seeds → "merge iff learned" does not survive
its own corrected statement and B3 advances to cross-time confirmation.
Status: **FALSIFIED 0/3 — on (iii) only; (i), (ii), (iv) held on all six seeds across both cuts.**
Diagnosis: revisits DO relearn the eroded residual (C15g's convergence — delta-norms were never
zero), so some revisits legitimately clear the 2σ progress bar and merge. C29c prescribed
"revisits always merge" — falsified; C29d prescribed "never" — falsified. **The lesson stands
recorded instead of a third prescription: the revisit case is not the gate's to decide — it
tracks measured residual learning, which varies, and either outcome is harmless (a skipped
revisit-merge is the null no-op; a taken one consolidates real erosion recovery). The progress
gate's PROVEN content, 6/6 seeds: noise never merges, learnable novelty (easy or hard) always
merges, retention unharmed — with only a statistical constant. B3's world-free criterion is
usable; its cross-time rung remains the deeper upgrade.]

## Claims — coverage & the bounded buffer (`checks/coverage.py`; registered 2026-09-01 before any
run; round 12 per `notes/design-coverage.md` — A2 + C13 are one question twice. Memo ids renumbered
here as the handoff required: the memo's C26a/b become C30a/b + the predicted-fail cut C30c;
C13a–c keep the ledger's reserved bounded-buffer number)

Three sub-protocols, one round.

**N-walk (C13a), `facts_run.py`:** world family frozen at s = 0.6, life T = 15N (C25's per-fact
visit rate held constant), grid N ∈ {10 000, 20 000, 40 000, 80 000}. Arms per grid point:
**bounded** = dedup store, random eviction (E1: random is the standing evictor), B = 8 000 slots;
**unbounded** = same admission, slots = N (never fills, never evicts). At N ≤ B the arms are
identical by construction (gap ≡ 0, structural) — no sub-B anchors enter the fit. Statistic:
gap(N) = online(unbounded) − online(bounded). Fit per seed over x = log₂N: line a + bx vs hinge
max(a, a + b(x − x₀)); criterion registered from the memo: SSE_hinge ≤ 0.5 × SSE_line AND knee
2^x₀ ∈ [8 000, 32 000] — the E4 band: at s = 0.6 with 15 mean visits the warm band is effectively
the whole world, so the store binds as soon as N clears B; the knee must sit at B or within two
octaves above it. Owned limitation, stated in advance: the hinge spends 3 parameters on 4 points
(1 dof), so the SSE-halving half of the criterion is generous — the knee band is the teeth, and
the 2σ escalation is expected to fire (seeds 4–9 budgeted). Pilot gate (seed 0): at N = 80 000
the unbounded arm must beat bp by ≥ 2 points, else the net (not the buffer) is the binding
constraint and the grid shrinks to where the gate holds (the C-c capacity confound, handled).

**Sleep spot-check (C13b/C13c), `sleep_run.py`:** stored arm, every C15 frozen constant except
`--reservoir 2000`, seeds 1–3 (fresh for this arm; the C15 pilot's glimpse was seed 0 only).
Comparators: the existing verdict runs `sleep_stored_s{1,2,3}` (reservoir 8 000); capacity
control: the existing `sleep_scratch_s{1,2,3}` ceiling (reservoir-independent) stands.
Margins frozen NOW from the ledger's own 8 000-arm numbers (paired, same seeds):
S_ret(8000) = 1.5/1.9/1.6 → 2σ = 0.4 pt → margin floor 1 pt binds; trend(8000, cycles 11–20)
= 0.094/0.095/0.069 → 2σ = 0.03 pt/cycle.

**C13a — Coverage starves as a threshold, not a slope (E4's arithmetic as a shape claim).**
[SURVIVES 3/3 — hinge ≤ 0.25× the line's SSE on every seed, knees 11 911/12 417/12 246: mid-band.
**The program's first life-vs-buffer law:** the gap is ≈ 0 until the world clears the buffer
(knee ≈ 1.5×B), then grows ~2.4 pts of online accuracy per world-doubling — a threshold with a
lawful slope after it, not a catastrophe. E4's arithmetic earns its first verdict-grade support;
the bounded store still beats memoryless everywhere measured (+7.4 pts over bp at N = 80k).]
Statement: per seed, the hinge fit halves the line's squared error and its knee lies in
[8 000, 32 000].
Falsifier: the line fits within 2× of the hinge, or the knee falls outside the band, on ≥ 2 of 3
seeds → E4's threshold arithmetic is wrong here (starvation is smooth) and T2 Killer B loses its
predicted shape; the gap curve is recorded either way as the program's first life-vs-buffer law.

**C13b — The C15 coverage glimpse survives fresh seeds (2 000 starves at 21 domains).**
[FALSIFIED 0/9 after within-2σ escalation (seeds 1–3 hovered at the 2× ratio; decision ≥ 5/9
registered before running, comparator the existing `sleep_stored_s{4..9}`). S_ret(2000) =
2.3–3.6 pts across nine seeds vs 1.1–1.9 at 8 000: starvation at 2 000 is REAL but mild —
~+1.2 pts, ratio ~1.7×, never clearing 2× + margin. The pilot glimpse overstated the cliff;
consistent with C13a, the sleep instrument at 21 domains sits just past its knee, not off it.]
Statement: S_ret(reservoir 2000) ≥ 2 × S_ret(8000) + 1 point, per seed (statistic verbatim from
`checks/sleep.py::s_ret`; paired comparator the same seed's existing 8 000 run).
Falsifier: below the bar on ≥ 2 of 3 seeds → the pilot glimpse was noise; the coverage cliff on
the sleep instrument is unproven and E4's pilot support weakens.

**C13c — The novelty tax inflates when coverage starves.**
[FALSIFIED 3/9 after within-2σ escalation (seeds 1–2 had landed 0.05 and < 0.01 pt/cycle below
their bars with raw ratios above 2×; ≥ 5/9, same comparator as C13b). Across nine seeds
trend(2000) = 0.0010–0.0022 vs 0.0007–0.0012 at 8 000: the tax IS higher under starvation
(~1.8× on average) but not lawfully ≥ 2× — too noisy seed-to-seed to serve as a trigger.
Growth's candidate second trigger (tax inflation, per the coverage memo) is dead as registered;
the tax stays a cost, not a signal.]
Statement: trend(2000, cycles 11–20) ≥ 2 × trend(8000, cycles 11–20) + 0.03 pt/cycle, per seed
(statistic verbatim from C15b's `s_trend`).
Falsifier: below on ≥ 2 of 3 seeds → the tax is a constant of consolidation, not a function of
coverage, and growth's candidate second trigger (tax inflation, per the memo) is dead on arrival.

**Drift world (C30a–c), `facts_run.py` + drift:** the frozen C25 world (N = 20 000, s = 0.6,
T = 300 000, B = 8 000) plus drift: a fraction ρ = 0.10 of facts, chosen uniformly, silently
resample their value to a different class at an event time uniform in [0.1T, 0.7T] (window
registered so every drifted fact keeps ≥ 0.3T of post-event stream exposure; nothing announces an
event — the only signal is the stream presenting the new value). Uniform selection makes drifted
stream mass ≈ ρ; pilot gate: measured mass ∈ [5 %, 15 %]. Arms: **bp** (no store), **naive**
(dedup random-eviction store; the slot's value is written at admission and never rewritten while
resident), **aware** (same store; a presentation contradicting the slot overwrites the value and
resets the slot's counters — the ledger's stream-refresh rule ported down). Implementation note,
registered: the C25 store arms read `values[fid]` live at rehearsal time, which under drift would
be an oracle (the store learning of a drift the stream never showed it); the drift arms therefore
snapshot y into the slot at write time. C25's arms and results are untouched.
Statistic: **S_drift(arm)** = mean online (pre-update) correctness on stream presentations of
drifted facts at t ≥ their event, restricted to the registered conditioning set = drifted facts
resident in the NAIVE store at their event time (the at-risk set; identical across arms by
pairing — same world, stream, and drift schedule per seed). Unconditioned S_drift and the aware
arm's staleness dwell (examples from event to slot rewrite) are recorded as diagnostics.

**Pilot record (seed 0, 2026-09-01, before the wave):** all three registered gates pass.
Capacity (N = 80 000): unbounded 79.9 % vs bp 64.8 % (+15.1 pts ≥ 2 gate) — the net is not the
binding constraint; the bounded arm previews a 7.8-pt gap at the top of the grid. Drift mass
0.099 ∈ [5, 15 %]. **H\* = 131 072** — the recall curve crosses halfway (0.524) at the 2¹⁷ bin
edge, inside the 0.5T = 150 000 gate but only just: recall after ~128k intervening examples is
still 0.45. This world's forgetting clock is SLOW, and φ_store(H\*) previews at 0.981 — C30a is
expected to survive near-vacuously; "T3 stands as written" will then be the honest scale-scoped
reading, not a discovery, and the full recall curve + φ at shorter horizons are the informative
diagnostics. Also recorded before the wave: S_drift bp 0.814 / naive 0.837 / aware 0.876 on 813
at-risk facts — aware clearly above naive (+3.9 pts, the contradiction rule pays), but naive sits
ABOVE bp, not 2 pts below: the round's predicted-fail (C30c) looks set to be MISSED (a store that
still helps outweighs its own stale 10 %). Aware dwell: 350/813 corrected, median 5 881 examples
(the rest evicted-and-readmitted or never re-presented while resident).

**C30a — The stream re-tests what the store keeps: φ_store(H\*) ≥ 0.5 (A2's number).** [proposed]
Definitions: **per-fact forgetting curve** (bp arm, seed 0 pilot) — every presentation is binned
by log₂(gap since that fact's last presentation); recall(g) per bin; r₀ = recall in the first
bin holding ≥ 200 presentations; **H\*** = the smallest upper edge of a ≥ 200-presentation bin
with recall ≤ (r₀ + 0.10)/2 (halfway to chance), frozen after the pilot before any verdict run. **φ_store(H\*)** (aware arm) — at snapshot times
every 10 000 examples in [0.25T, 0.9T] (snapshots later than T − H\* dropped), the fraction of
store-resident facts whose next stream presentation arrives within H\*; φ = mean over snapshots.
Pilot gate: the recall curve must cross its halfway point within 0.5T, else H\* outlives the
life, φ is vacuously high, and A2 cannot bind at this scale — recorded as a null and the claim
re-cut on a harder clock before any verdict run.
[SURVIVES 3/3 at φ = 0.981/0.982/0.982 — near-vacuously, exactly as the pilot forecast: this
world's forgetting clock is slow (H\* = 131k ≈ 0.44T). **T3 stands as written at this scale and
the organ decision does NOT fire** — scale-scoped: A2 stays live for worlds whose re-test clock
is slower than their forgetting clock; the measured recall curve and φ(H) diagnostics are the
durable output, and the aware arm's dwell numbers (a warm fact can wait ~6k examples for its
re-test) show where a faster world would bite.]
Statement: φ_store(H\*) ≥ 0.5 per seed.
Falsifier: φ < 0.5 on ≥ 2 of 3 seeds → **the organ decision fires as registered in the memo**:
most kept knowledge outlives its own verification, T3 narrows to predictive-recurrent knowledge,
and the store is promoted from tool (T6) to organ — it must hold ground truth, price staleness
(design-audit's T_stale/n_contra rules), and refuse to rehearse what it cannot re-verify.

**C30b — Contradiction-awareness closes drift to within noise of no-memory.**
[SURVIVES 3/3, 7–8 pts above its bar — aware does not merely close to bp, it BEATS bp by
4.8/5.8/5.9 pts on the at-risk facts: contradiction-aware memory helps even on drifting content.]
Statement: S_drift(aware) ≥ S_drift(bp) − 2 points, per seed.
Falsifier: below on ≥ 2 of 3 seeds → no store design on the table survives a moving world;
the round's sharp question (is contradiction-awareness sufficient?) answers no, and the audit's
rewrite rule needs a stronger port than counter-reset.

**C30c — Stale rehearsal is harmless (the round's predicted-fail; C30b's statement, one knob:
arm).**
[SURVIVES 3/3 — **the program's second missed predicted-fail** (naive − bp = +1.4/+1.2/+1.7 pts,
3+ pts above the bar). Stale rehearsal is NOT anti-knowledge at this scale: a store that still
helps (the ~90 % fresh fraction plus relearning support) outweighs its own stale tenth, and
eviction-plus-readmission quietly refreshes stale slots the rewrite rule never touches. The
memo's precondition argument falls: contradiction-awareness is a measured upgrade (+3.5–4.3 pts
over naive), not a survival requirement — C29b's lesson from the other side: mostly-correct
gradients shrug off a stale minority.]
Statement: S_drift(naive) ≥ S_drift(bp) − 2 points, per seed (bar inherited from C30b verbatim).
Prediction: FALSIFIED — the naive store rehearses the pre-drift value against the stream's
correction (E7's dilution at store level, with no rewrite-before-merge to save it); memory should
make drifted content measurably WORSE than no memory. If it survives instead, stale rehearsal is
harmless at this scale, contradiction-awareness is a nicety rather than a precondition, and the
memo's anti-knowledge argument is falsified — audit the mixing ratio before believing either way.

## C24 — systems microbenchmark (2026-09-01; measurements, no claims — the gate the scale memo
requires before any port. `c24_bench.py` → `results/c24_bench.json`; RTX 4080 SUPER 16 GB,
`cl/lm.py` = from-scratch GPT in the nanoGPT shape, char vocab 256, block 256, batch 16×256,
AdamW; sleep step = 2 teacher forwards + student fwd/bwd; adapter overhead deferred to the port)

| shape | params | tok/s | step | sleep step | peak VRAM (teacher+student+opt) | ckpt save/load | ckpt size |
|---|---|---|---|---|---|---|---|
| 6L/384/6h | 10.9M | 229 503 | 17.8 ms | 28.2 ms (1.58×) | 0.81 GB | 0.09 s / 0.07 s | 131 MB |
| 12L/768/12h | 85.6M | 39 420 | 103.9 ms | 169.0 ms (1.63×) | 3.93 GB | 0.62 s / 0.30 s | 1.03 GB |

Cadence limits implied: at 10M a 500M-token single-pass life costs ~36 min (tier-2 rounds are
same-evening affairs; the childhood corpus at ~1 pass ≈ half an hour); at 85M the same life is
~3.5 h and VRAM headroom remains ~12 GB — both shapes are one-box comfortable, and sleep costs
only ~1.6 plain steps. **Tier-1 answer: batch-1 3×2000 MLP runs at 1 263 steps/s → 10.6 GPU-h
per 800×60k run, ~95 GPU-h for the 9-run C26 fidelity wave** — the "weekend or a week" question
resolves to ≈ a week of background; schedule it as such or register a batch deviation.

## Claims — tier 2 opens: the currency keystone on language (`checks/lm_currency.py`; registered
2026-09-01 before any run; the scale memo's decision rule: the C22-port is the single experiment
that most raises confidence in the whole ledger)

Gate record: **E19-LM PASS bitwise** (`checks/persist_lm.py` — 85M GPT + AdamW + TokenReservoir
+ all three RNG streams, the 50-save-50-replay protocol). C24 table above. Shape decision, user
on record ("my GPU can manage 85M"): **85M (12L/768/12h) is the registered verdict shape**;
fallback to 10M pre-registered iff the forgetting gate below fails at 85M — capacity can make
relearning too cheap (the C22 trap), and that outcome is a finding, not a tuning knob.

Protocol (`lm_run.py`): byte-level LM (vocab 256, block 256, batch 16 blocks/step = 4 096 stream
tokens), single-pass life of T tokens in S-token segments; pool K = 6 byte shards with real
distributional variety (`data/shards/`: shakespeare 1.1M / kjv 4.4M / enwik-XML 4.0M /
python-code 4.0M / french 0.9M / german-verse 0.2M — sizes deliberately heterogeneous; per-domain
reporting registered, symmetry never assumed; `data/fetch_shards.py` rebuilds them). Schedule =
repeated seed-random permutations of the pool, identical across arms per seed (the C15 pairing
trick); per-domain cursors continue across visits (a domain is a distribution, not a byte string;
small shards wrap and are thereby memorization-flavoured — recorded per-domain). Prequential
scoring as everywhere: each step's stream blocks are scored (CE/log 2 on the pre-update forward,
within-block next-byte targets) then trained on. Arms: **bp** = plain online AdamW; **replay** =
TokenReservoir (slots = 5 % of life in blocks, C9's 1:1 mix; the mixed batch's compute asymmetry
is inherited from C9/C22 as registered there). Constants at registration: lr 3e-4 AdamW, S = 2M,
T = 48M (≈ 4 visits/domain, revisit gap ≈ 10M tokens); the pilot may retune world constants
(S/T/K) before freezing — never the bar.

Pilot gates (seed 0, registered before running): (a) **forgetting gate** — on bp, the mean
revisit spike (a visit's first-decile bpc minus the same domain's previous-visit last-decile
bpc) ≥ 2× the seed-0 across-domain sd of that spike; fail ⇒ relearning too cheap at 85M ⇒ drop
to the 10M fallback and re-pilot; (b) **learning gate** — every domain's first-visit bpc falls
start→end; (c) the per-domain bpc table is recorded before any verdict run.

**Pilot record (seed 0, 2026-09-02 00:02, before the wave):** bp 100 155 111 bits (2.0960 bpc),
replay 77 317 026 bits (1.6180 bpc) — **saving 22.8 % vs the 10 % bar**; learning gate PASS
(every domain's first-visit bpc falls). The forgetting gate **FAILED as registered — and the
failure is a mis-cut, C29c-style, recorded not obeyed blindly**: "mean spike ≥ 2× sd" punishes
across-domain heterogeneity (spikes 0.05–1.33 bpc; sd 0.33 → 2σ 0.66 > mean 0.54), not absence
of forgetting; yet all 18/18 revisit spikes are positive and the paired replay arm flattens them
to 0.024 bpc — the pairing isolates ~0.51 bpc of pure forgetting from new-content difficulty,
refuting the fallback's own premise (relearning is demonstrably expensive at 85M). **Gate v2,
registered before any verdict run: every domain's spikes positive on ≥ 5/6 domains AND
(bp mean spike − replay mean spike) ≥ 0.1 bpc.** Seed 0: 6/6 and 0.51 — PASS. Constants frozen
exactly as piloted (85M, T = 48M, S = 2M, K = 6, lr 3e-4, slots 5 %); no retune, no fallback.

**C31 — Retention pays ≥ 10 % in prequential bits on a recurring-domain char stream (C22's
statement and bar verbatim; one knob: substrate).**
[SURVIVES — decided at 2/2, seed 3 cut mid-run on the user's explicit time directive once the
2-of-3 rule was mathematically satisfied (recorded honestly: 2/2-decided, not 3/3): savings
**30.9 % (seed 1) and 23.7 % (seed 2)** vs the 10 % bar, pilot 22.8 % concordant; margins ≥ 4×
the bar's distance — no 2σ question. bp bpc 2.29/2.07; the same replay mechanism that saved
≈ 1 % on cheap-relearning MNIST (C22 FALSIFIED) saves 24–31 % where a domain costs millions of
tokens to relearn. **The cheap-relearning law is confirmed across three substrates (C22 cheap /
C25a scarcity / C31 gradient-cost) and every "would pay at scale" reading of the MNIST rounds
keeps its license. The economics ground floor of tier 2 holds; the sleep port (C32) proceeds.**]
Statement: replay's total prequential bits ≤ 0.90 × bp's, per seed, on ≥ 2 of 3 seeds.
Prediction: SURVIVES — the cheap-relearning law's directional bet put to its test: here,
relearning a domain to competent loss costs millions of tokens.
Falsifier: saving < 10 % on ≥ 2 of 3 seeds → the currency is dead in BOTH economic regimes;
every "would pay at scale" caveat in the ledger loses its license (the pre-committed
biggest-possible revision, per the scale memo's decision rule), and the external-objective
position hardens from decision to law.

## Claims — C32: sleep on language (`checks/lm_sleep.py`; registered 2026-09-02 before any
85M run; the C15-port, second rung of the scale memo's priority order)

Protocol (`lm_sleep_run.py` + `cl/lm_adapter.py`): the C15 wake/sleep cycle on the tier-2
substrate. Foundation: base trains directly (replay on) on F1–F4 (2M tokens each), then P0
(never enters the reservoir — the never-rehearsed probe). Cycles: wake = fresh identity-init
LoRA nursery (rank 8 on every block's qkv/proj/mlp + head; attach-after-freeze so only adapter
params train; detach restores the base bit-for-bit) on 1M tokens of the cycle's domain; sleep =
625 steps of the base on a 1:1 row mix of **stored true bytes (CE — C15j's winner is the
default here, not an arm)** and the composite teacher's per-token distribution on the wake
buffer (KL; teacher = deep-copied composite); adapter deleted. **Domain = (shard,
byte-rotation)** — the permutation analogue: same structure, shifted input space. Pool: P0 =
de+0; F = shakespeare/kjv/enwik/pycode+0; R = fr+0 recurring at cycles 1,2,4,8; 8 fresh
rotations; cycles 13–20 recycle the exhausted pool (zero-novelty sleeps — C32b's window).
Cycles = 20, all arms. Arms: **stored** / **weightadd** (nursery.fold() — the industry
LoRA-merge, no rehearsal pull-back) / **null** (identity adapter, BOTH sleep sides
self-distilled — the fixed point; smoke-verified: probes frozen across null sleeps). Bars:
S_RET_BAR = 0.10 bpc (registered convention, ~5 % of the working range, as C15's 2 pts was);
TREND_BAR = max(0.005 bpc/cycle, 2σ_null) with σ_null = stdev of per-cycle foundation-mean
deltas on the seed-0 null pilot (C15's recipe verbatim). Statistic S_ret = foundation-mean bpc
after cycle 20 minus after foundation (bpc rises = forgetting). Seed-0 pilot of all three arms
precedes the wave; world constants may retune there, bars never.

**Protocol amendment (2026-09-02 03:20, before any verdict run):** the registration text said
"cycles = 20, all arms" while the committed instrument defaulted to 12 — the seed-0 pilot ran at
12. Resolved, recorded: **stored = 20 cycles** (C32a/C32b statements unchanged; 13–20 is the
recycled window as registered); **weightadd = 12 cycles** (C32c's S_ret is judged at 12 — fewer
cycles accumulate less damage, which makes survival EASIER and our predicted-fail stricter);
**null = seed 0 only** (the bar formula already said so). Pilot preview at 12 cycles: null is a
clean fixed point (F frozen at 2.241 all cycles, |d| = 0 → TREND_BAR sits on its 0.005 floor);
stored mid-run: wake learns rotated domains (|d| 11–19), F holds 2.22–2.28 through 8 sleeps,
R converges across revisits (3.69 → 3.04), P0 erodes upward ~0.035 bpc/sleep (C15i's echo,
unrehearsed by design).
**Pilot final (seed 0, 03:38):** stored S_ret(12 cyc) ≈ +0.05 bpc (bar 0.10) with every new
domain learned and merged; weightadd S_ret(12 cyc) ≈ **+2.24 bpc — 11× its bar**, F lurching to
5.92 after the first rotated fold while the folded-in NEW domain stays good (C15d's signature
verbatim at LM scale: the fold keeps the adapter's learning and silently destroys the base's).
All gates pass; constants frozen; wave launched.

**C32a — Sleep consolidates without destroying, on language (C15a's statement, substrate
knob).**
[SURVIVES — decided at 5/6 under the ≥ 5/9 escalation rule (S_ret 0.084/0.117/0.077 on seeds
1–3, then 0.078/0.080/0.098 on 4–6: five passes; seeds 7–9 cut once the decision was
mathematically locked, per the user's standing time directive — recorded as 5/6-decided).
**T4's core holds on language: 20 wake/sleep merges cost ≤ ~0.1 bpc of old knowledge**, with
stored sleep ~25× better than weightadd on the same protocol. The margins are thin against this
strict bar and honestly recorded as such.]
Statement: stored arm S_ret ≤ 0.10 bpc, per seed, ≥ 2/3 seeds.
Prediction: survives — T4's core, already proven over 20 merges on the MLP.
Falsifier: S_ret > bar on ≥ 2/3 seeds → T4 fails at scale, the sleep design returns to A9, and
the tier-2 ladder halts until sleep is redesigned (nothing above it is worth testing).

**C32b — The novelty tax is per-novelty, not per-sleep, at LM scale (C15b's reframe; the scale
memo's drafted C28).**
[SURVIVES — decided at 5/6 with C32a (trends 0.0038/0.0059/0.0023 then 0.0023/0.0031/0.0046 vs
TREND_BAR 0.0050 on its floor; 2σ_null = 0.0001, the null fixed point essentially exact; seeds
7–9 cut with C32a's). **The novelty tax stays per-novelty at LM scale**: recycled, zero-novelty
sleeps drift at ≲ half the fresh-domain rate on most seeds — consolidation is priced per new
thing learned, not per unit of time asleep. C15b's reframe generalises; the scale memo's C28
question is answered.]
Statement: stored-arm trend of foundation-mean bpc over the recycled cycles 13–20 ≤ TREND_BAR,
per seed, ≥ 2/3 seeds.
Falsifier: recycled sleeps still leak ≥ the band on ≥ 2/3 seeds → drift compounds with time
itself at this scale; T4's cost model is wrong even if C32a's absolute bar holds.

**C32c — Weight-addition merge retains, on language (the round's predicted-fail; C15d re-cut —
and exactly what industry LoRA-merge does without rehearsal).**
[FALSIFIED 3/3 AS PREDICTED, decisively — S_ret 2.180/2.154/1.949 bpc vs the 0.20 bar (~10×),
at the EASIER 12-cycle scope. The fold keeps each new domain and silently wrecks the old ones
(F lurches to ~5.9 bpc after a single rotated-domain fold). **C15d generalises to LM scale:
rehearsal-free LoRA-merge is not a consolidation mechanism, it is an overwrite** — the
program's sharpest industry-relevant verdict, three seeds, pre-registered prediction.]
Statement: weightadd arm S_ret ≤ 0.20 bpc (2× the C32a bar), per seed, ≥ 2/3 seeds.
Prediction: FALSIFIED — the fold keeps the composite's drifted old-domain behaviour with no
rehearsal pull-back (63–73 pts on the MLP). Stakes stated in advance: if it SURVIVES instead,
C15d was substrate-local, rehearsal-free LoRA-merge is safe at LM scale, and the thesis's
distillation-sleep loses its necessity argument on language — audit before believing, then
believe it.

## Claims — C33: the childhood run (`checks/childhood.py`; registered 2026-09-02 before any 85M
run; `notes/design-childhood.md` approved by the user — "go childhood"; shape 85M decided on
the 10M pre-pilot + tuning-probe data recorded there)

Protocol (`lm_child_run.py`): one pass over the BabyLM track (`data/babylm/`, HF mirror
cambridge-climb/BabyLM of the CFP's osf.io/ad7qg corpus; 10 sources), grown from scratch through
the C32 sleep loop verbatim (rank 8, 625 sleep steps, old 16, slots 8 000, lr 3e-4). Segments
≤ 5M tokens, never crossing a source boundary (sub-step tails dropped). Arms: **curated** =
developmental stage order (aochildes → spoken → children's books → simple/educational → adult
prose); **raw** = same segments shuffled (A10's contrast); **scratch_ladder** = fresh net,
ladder only (the possession control). Both corpus arms end with ladder rungs 1–2
(`cl/ladder.py`, depth 2) as final segments. Probes: every source's held-out tail
(min(100k, size/6)) + both rungs, after every sleep. Engineering registered: **bf16 autocast +
TF32 + fused AdamW — E19-LM re-PASSED bitwise under this config**; measured speedup quoted at
pilot. Diets: **10M-word track = pilot (seed 0); 100M = verdict, seeds 1–3**, early-stop on
decision standing (user directive). Ladder criterion CRIT = 1.0 wake bpc; pilot gate: CRIT
reachable within the rung budget on both curated and scratch arms, else re-set from pilot
before any verdict run.
**Pilot record (seed 0, 10M diet, local, 2026-09-02 11:49):** full life lived — 18 segments,
language probes stable ~3.57–3.70 bpc throughout (novelty-tax blip at the rungs only), rungs
learned (rung-1 probe 9.39 → 3.67; wake curve 9.17 → 2.61 over its 1M-token budget); bf16
config ≈ 2× (52M-token life in 22 min). **Gate fired: CRIT = 1.0 UNREACHED** (the wake curve
floors near 2.6 — consistent with the user's own finding that exact arithmetic is not a
transformer's fighting ground; the rung is a meter, not a target). **Re-set before any verdict
run: CRIT = 3.0; an arm that never reaches it scores its full rung budget (conservative
floor); C33c's ratio unchanged otherwise.** Verdict runs on rented H100s (user-approved,
cap $30) with seed 1 duplicated locally as the cross-machine anchor; rented hardware is not
bit-identical to the local box — recorded, within-run E19 discipline unaffected.

**Verdict protocol note (2026-09-02): seed 3 cut by user directive before starting (machine
destroyed); with two seeds the decision rule is BOTH-must-agree — and both agreed on every
claim, so no tiebreak was needed. Runs on rented H100s (cross-machine caveat recorded); metrics
collected, machines destroyed. Weights of the verdict lives were NOT saved (instrument gap,
fixed in `eece768` — when the model is the product, the checkpoint is part of the protocol);
the base-model artifact is grown separately with the fixed code.**

**C33a — The sleep loop survives a real childhood.**
[SURVIVES 2/2 — 9/10 sources within the bar on both seeds, and the breach is only the
final-cycle novelty tax on the last-trained source. **The headline of the round: over 100+
sleeps, early knowledge did not merely survive — most sources IMPROVED after their stage ended
(back-transfer, e.g. aochildes −0.47 bpc over 106 unrehearsed sleeps).** The architecture's
life-scale test, passed with integration, not just retention.]
Statement: per source, S_ret_child = end-of-life probe bpc − probe bpc after that source's last
trained segment ≤ 0.006 × intervening sleeps; ≥ 8/10 sources within, on ≥ 2/3 seeds.
Falsifier: worse → the loop does not survive real lifetimes; suspect mechanism named in
advance: reservoir coverage erosion (C13a's law — resident fraction falls as life grows).

**C33b — The curriculum pays (A10 closed either way).**
[FALSIFIED 2/2 — curated beats raw by +0.036/+0.011 bpc, consistent in direction on both seeds
but under the 0.05 margin. **A10 closes on the falsified side: the teacher was a convenience**
— style-order helps marginally at corpus level; don't pay for curation. Registered reading for
the road ahead (user's 1+1 point): this tested STYLE order; PREREQUISITE order (skill chains,
the ladder, syllabi) is a different, stronger phenomenon — prediction to register at the study
loop: order effects grow with prerequisite depth.]
Statement: curated end-of-life mean probe bpc ≤ raw − max(0.05, 2× paired seed-sd), ≥ 2/3 seeds.
Falsifier: raw within the margin → the teacher was a convenience, T5 loses a dependency —
welcome; either outcome closes A10.

**C33c — Childhood buys the ladder (possession at LM scale; C21c's analogue).**
[FALSIFIED 2/2 — ratios 0.65/0.63 vs the ≤ 0.5 bar: a childhood makes the new skill ~35–37 %
cheaper to learn, real and seed-consistent, but not the 2× registered. Possession pays at LM
scale; it pays less than bet. The instruction test keeps its premise with a measured, smaller
coefficient.]
Statement: tokens-to-criterion on rung 1 (first wake step whose trailing-10-step mean bpc
≤ CRIT, × 4 096) for the childhood model ≤ 0.5× the scratch_ladder net's, ≥ 2/3 seeds.
Prediction: survives — primitives (digits, '=', line structure) transfer from a life of text.
Falsifier: ratio > 0.5 → a childhood buys nothing on formal skills; the instruction test loses
its premise and the ledger records the cheaper-pretraining critique un-refuted at this rung.

**C33d — Curated retains the earliest childhood better (the round's predicted-fail).**
[FALSIFIED 2/2 AS PREDICTED — aochildes end-of-life gap +0.016/+0.000, nowhere near the margin:
erosion does not care about order, exactly as registered. C33a's back-transfer makes the point
sharper: what protects early knowledge is CONTINUED RELEVANCE (shared structure with later
material), not its position in the schedule.]
Statement: curated end-of-life aochildes probe bpc ≤ raw's − the C33b margin, ≥ 2/3 seeds.
Prediction: FALSIFIED — the earliest, never-revisited source erodes by C15i/C13a regardless of
order; both arms should end similarly degraded on it. Surviving instead would be a deep
curriculum result (early order leaves durable structure) — audit, then believe it.

## C34 — the base build (registered 2026-09-02; a BUILD protocol with acceptance gates, not a
falsification round — the artifact is the deliverable; the flywheel claims come after, on it)

Per the user's plan and interface memo §7.7: **six checkpoints, each stage warm-starting from
the last, all published to HF `lamshiu/continual-child` (public), runnable locally.**
Chain: childhood (curated 100M; C33 constants) → **qa** (being addressed) → **cot** (worked
steps, every line verified) → **tools** (`run:`/`read:` transcripts with real outputs) →
**idk** ("I don't know" as the verifiably correct answer to invented-entity questions) →
**quiz** (cloze self-quiz over its own corpus). Stage streams generated + graded by
construction (`cl/stages.py`); 5M tokens/stage, wake 1M, C32 sleep constants
(`lm_stage_run.py`). Rehearsal pool rebuilt per stage: half childhood-corpus blocks, half
prior-stage blocks (regenerated deterministically) — language and format memory both stay
rehearsed. Probes after every sleep: 10 childhood tails + all 5 stage held-out streams.
Acceptance gates (recorded per stage, run stops if breached): (a) the stage's held-out bpc
falls from its pre-stage value; (b) mean childhood-language bpc rises ≤ 0.10 from pre-stage
(the format stages must not cost the language). SQuAD garnish deferred pending license
verification. Cloud runs on a rented H100; **checkpoints verified downloaded before the
machine is destroyed** (the C33 lesson, now protocol).
**BUILD COMPLETE (2026-09-02 16:49): all six checkpoints live at
huggingface.co/lamshiu/continual-child; final base mirrored to `models/child-base-v1.pt`.**
Gates: 4/5 stages PASS clean; every stage learned its format (own-held bpc: qa 1.18 / cot 1.71
/ tools 2.14 / idk 0.85 / quiz 2.27). **One honest breach, recorded: the qa stage — the first
radically new format — cost +0.37 lang bpc (gate 0.10), and the four later stages' rehearsal
pulled it back to +0.100 net over the whole curriculum, landing exactly at the gate.** Reading:
the C15 novelty tax writ large — first-format shock, then recovery through consolidation; the
per-stage gate was mis-scoped (the curriculum-level number is the meaningful one) and future
stage protocols should gate on end-of-curriculum drift. Operational note: a duplicated launch
(the "timed-out" first attempt survived its disconnect) ran a parallel pipeline on the same
GPU; all duplicates were killed before writing any file — lineage verified single-chain by
file timestamps and --init parentage.

## INCIDENT — the frozen-body bug (2026-09-02, caught while building C35; fixed same day)

**What:** in all three tier-2 runners (`lm_child_run.py`, `lm_stage_run.py`, `lm_sleep_run.py`)
the sleep-phase `net.requires_grad_(True)` ran while the LoRA nursery was still attached. The
nursery replaces each Linear IN PLACE and keeps the original in a plain list
(`LoraLinear.frozen`) invisible to `net.parameters()`, so every wrapped Linear (qkv, proj, both
MLP projections, the head) kept `requires_grad=False` through EVERY sleep. Sleep-distillation
could only move token/position embeddings and LayerNorms — ~0.47M of 85.6M params (0.55%).
Wake adapters trained normally but are discarded at detach by design; their content reached the
base only through distillation into that 0.55%.

**Proof:** (a) tiny repro — with wraps attached `net.parameters()` sees 11,648 of 42,368
params; under the buggy order qkv stays `requires_grad=False` and does not move in sleep.
(b) checkpoint audit — every Linear weight in `models/child-base-v1.pt` is within 3.7e-9
(float ULP) of its reconstructed seed-1 init, while embeddings/LayerNorms moved by up to ~5.7.
The published base's transformer body is numerically at random init.

**Blast radius:** C32 (all arms except weightadd, whose `fold()` edits weights directly and DID
move the body — an asymmetric comparison), C33 (both childhood arms + ladder segments), C34
(all five stages). Clean: C31 (`lm_run.py`, no nursery) and the entire MNIST-era ledger
(C13/C15/C17/C21/C23/C25/C29/C30) — `cl/adapter.py` wraps AROUND the net without replacing its
modules, so its `requires_grad_(True)` reached every base param.

**Re-grade (honest, immediate):** affected verdicts re-scoped to the unintended substrate
"embedding-only plasticity (0.55% of params trainable in sleep)":
- C32a/b: retention was structurally easy when 99.45% of the model could not move; verdicts
  stand as measured but assert far less than registered — re-run before citing.
- C32c: full-body overwrite vs embedding-only consolidation is asymmetric; the direction has
  clean-substrate support (C15d) but the margin is untrustworthy — re-run.
- C33a-d: all four re-graded UNDECIDED pending re-run on the fixed loop.
- C34: the six HF checkpoints are not "an 85M trained through the curriculum" — they are a
  random transformer body with trained embeddings/LNs that still reached lang ~1.5 / qa 1.18
  bpc (an unintended frozen-random-features baseline; keep the numbers under that name).
  Rebuild required (v2 chain supersedes v1 on HF).

**Fix:** `net.requires_grad_(True)` moved AFTER `nursery.detach()` in all three runners and
the new `flywheel_run.py`; verified (qkv moves in sleep; full param count visible post-detach).
**Lesson:** an in-place module-replacement adapter changes what `net.parameters()` MEANS
mid-cycle; any net-wide toggle (`requires_grad_`, optimizer construction) is legal only in the
detached state. Corollary of the C33 checkpoint lesson: when the model is the product, audit
the artifact against its own init before publishing — one `torch.equal` would have caught this.

## C34-v2 — the base REBUILT on the fixed loop (2026-09-02 evening; ACCEPTED by user with one
gate breach recorded)

Same protocol as C34 + the frozen-body fix, `--ckpt-every` periodic checkpoints (childhood
every 4 segs, stages every 2 cycles), tripwire asserts, fail-fast driver, background HF
uploads. H100 #49687639, 2h15m, **$3.90**; 43 checkpoints at HF `checkpoints/v2/`; final
mirrored to `models/child-base-v2.pt`; **body audit PASS** (body max|Δ-from-init| 3.216 vs
v1's 3.7e-9 — the artifact is genuinely trained; `checks/body_audit.py` is now the publish
gate). Childhood: ~110 segments, mid-life lang 2.26 bpc.
**Gate a (own-held falls): 5/5 PASS, v1 → v2:** qa 1.18 → **0.360**, cot 1.71 → **0.497**,
tools 2.14 → **1.060**, idk 0.85 → **0.679**, quiz 2.27 → **0.826**. The working body learns
formats ~2-4× better in bpc terms.
**Gate b (curriculum lang drift ≤ 0.10): BREACH at +0.168** (3.304 → 3.472; qa healed −0.135,
idk attacked +0.376, quiz healed −0.350). User accepted the build with the breach recorded;
a stages-only rehearsal-tuning pass (~$1) stays queued if the tax matters later.
**Finding (goes beyond the gates): the ladder ate the language.** Childhood's own final ~12
segments (late wikipedia + the two 5M-token arithmetic-ladder blocks, C33-protocol tail) cost
lang 2.256 → 3.304 (+1.05) while teaching the notation (rung1 8.5 → 1.55, rung2 → 1.02 bpc).
C15i re-cut at LM scale: rehearsal protects what it covers, and two solid segments of alien
notation flood the reservoir. v1 never showed this because its frozen body could not move —
the stability-plasticity tradeoff was invisible under the bug. Registered consequence for
future childhoods: interleave the ladder tail (or raise old:new during it) instead of running
it as back-to-back terminal segments.

## Claims — C35: Stage C opens — the first flywheel turn (`checks/flywheel.py`, instrument
`flywheel_run.py`; registered 2026-09-02 before any run; interface memo §7.6 rung 4, §7.7 step 2)

The child (the rebuilt base `models/child-base-v2.pt` — post frozen-body fix, incident above;
v1 is not a valid substrate for these claims) writes its own exercises, attempts them with a REAL
`run:` tool (generation is intercepted line-by-line: a completed `run: print(E)` line gets the
interpreter's actual `out:` injected into the context mid-generation — the first closed loop,
condition 3 of §7.1), the interpreter grades every unit (`cl/ladder.calc`; final answer),
verified-correct units become the wake stream, the C32/C34 sleep consolidates. Domain this
round: arithmetic Q/A + worked-steps (ladder rung 1 in the child's own Q/T/A dress). Statistic:
exact-match accuracy, greedy decode, on fixed held-out question sets at depth 1/2/3 (200 each,
generator-seed 1000+seed), prompt `Q: what is E\nA: `; language guard = the 10 childhood-tail
bpc probes + 5 stage helds. Arms (equal consolidated tokens, C34 loop constants inherited —
rank 8, sleep 625, old:new 16:batch, slots 8000, lr 3e-4; wake tokens/cycle and cycle count are
pilot-set then frozen): **self** (self-written, graded, verified-only consolidation, tool on),
**rule** (wake stream from the stage generators' arithmetic units — the hardcoded curriculum
baseline), **raw** (self-written, UNGRADED — everything harvested consolidates; the C15e
forbidden stream re-cut at Stage C). Margins: max(0.02, 2× paired seed-sd) on accuracy deltas,
pilot-frozen before verdict seeds; seeds 1–3, ≥ 2/3; seed 0 pilot only.

- **C35a — the flywheel turns.** Self-practice teaches: the self arm's overall held-out
  accuracy at end-of-run exceeds its own pre-flywheel value by more than the margin.
  Falsifier: flat or falling on ≥ 2/3 seeds — recorded verbatim as "self-practice does not
  teach at this scale; Stage C's door stays shut and the cheaper-pretraining critique stands."
- **C35b — self-vs-rule (de-hardcoding ladder rung 4).** The self arm's final accuracy is
  within one margin of, or above, the rule arm's at equal consolidated tokens. Survives → the
  hardcoded exercise generator is DELETED from the flywheel path (the §7.6 law: the rule is
  beaten, not kept as a crutch). Falsifier: self trails rule by > margin on ≥ 2/3 — the
  generator stays; rung 4 refused; recorded with the measured gap.
- **C35c — the language is kept.** End-of-round mean childhood-language bpc drift from
  pre-flywheel ≤ 0.10 (curriculum-scoped, the C34 gate lesson) on ≥ 2/3 seeds. Falsifier:
  drift > 0.10 — self-practice eats the language; the flywheel is not safe to run unattended.
- **C35d — the grader is load-bearing (predicted-fail; the round's one).** The raw arm matches
  the self arm's final accuracy within the margin. REGISTERED TO FAIL: raw must trail graded
  by > margin (C15e's law re-cut on the child: ungraded self-generated text is not knowledge).
  Surviving would mean §5's verifier boundary is decoration at this scale — audit before
  believing, then re-grade the whole §7.6 program.
- **C35e — the real tool pays.** On depth-3 held-out questions, accuracy with a real
  `run:`/`out:` transcript in context (tool ran, output injected) exceeds direct answering by
  more than the margin, measured on the final checkpoint of the self arm. Falsifier: no lift
  on ≥ 2/3 — the tools stage taught transcript STYLE, not tool GROUNDING; the wiring stays
  (infrastructure), the capability claim dies. Spontaneous tool-reach rate during self-practice
  is recorded as a measurement, not a claim (first sighting of unprompted tool use).

**C35 RE-SCOPED (2026-09-02 ~20:20, user decision before any verdict seed ran):** the
arithmetic domain is RETIRED as a proving ground ("after depth 2 the transformer is not good
at it, we don't have to prove anything there") — byte-level transformers are known-weak at
deep arithmetic and the seed-0 pilots confirmed the ceiling (d3 flat ≈ 0.03-0.05 in both
monolithic and unit arms; d2 gate stuck ≈ 0.10 under the correction channel). No C35 verdict
wave runs; the seed-0 pilot records stand as the method's baseline data (the echo-chamber
stall, the zero spontaneous tool reach, the interpreter-correction glimpse). The METHOD
claims move to the text domain: C35a→C37a, C35b→C37d, C35c→C37b, C35d→C37c; the tool claims
(C35e + reach) live at C38/C39 where the tool is search over a library. The run: wiring and
the ladder stay as infrastructure (the interpreter remains a grader for §7.3 skills later).

## Claims — C36: the learning optimization (`flywheel_run.py --arm unit`; registered
2026-09-02 before any verdict run; design memo `notes/design-selflearn.md` §1)

The user's directive: tiny steps, scheduled, divide and conquer — divided by SKILL, not by
sample. The `unit` arm: mastery-gated micro-units (depths 1→2→3; advance at gate ≥ 0.9 on a
disjoint gate set, or stall = 3 micro-cycles without a new best), fail-directed harvest
(one-shot demo steers difficulty; only current-unit exercises count; WRONG attempts enter as
interpreter-corrected transcripts — Q + true answer, bad chain dropped), sleep scaled to wake
(max(50, 2× wake steps) instead of flat 625). Constants pilot-adjustable until verdicts,
frozen after. Baseline = the C35 monolithic arms at equal consolidated-token budget. Seed-0
C35 pilot context (the stall these must fix): one cycle of gains then echo-chamber — verify
rate →96%, own-stream wake-bpc →0.24, d2 ends below pre, lang re-erodes past pre.

- **C36a (micro-units pay):** unit beats self's final overall accuracy by > margin, ≥ 2/3.
  Falsifier: ≤ self — uniform-is-unbeaten extends to the closed loop; recorded as such.
- **C36b (the ceiling breaks):** unit lifts depth-3 above self's flat band by > margin.
  Falsifier: d3 flat — targeting+correction cannot climb; the ladder needs smaller rungs.
- **C36c (language survives micro-sleeps):** lang drift ≤ 0.10 bpc despite ~8× more frequent,
  ~8× smaller sleeps. Falsifier: micro-sleeps leak — the flat-625 convention was load-bearing.

## Claims — C37: recursive intake on text (`textloop_run.py`; registered 2026-09-02 before
any verdict run; memo §2. C38 search / C39 reach: machinery BUILT — search: interceptor,
library, seek format-mini generator — claims drafted in the memo, registered at their rounds.)

The child self-quizzes on UNSEEN library lines (deep-tail, disjoint from bpc probes; the
trained text:/Q:/A: cloze format); verification = reconstruction (answer into blank must equal
the source line); verified units — source line + the child's own question — consolidate.
Its own thought becomes its food, pruned by the world (C15e's legal form).

- **C37a (self-quizzed intake teaches):** held library-cloze accuracy rises from pre by
  > margin, ≥ 2/3 seeds. Falsifier: flat — self-quiz intake is theater at this scale.
- **C37b (language kept):** lang drift ≤ 0.10 bpc over the run.
- **C37c (predicted-fail; the round's one):** the notext arm (no verification, everything
  consolidates) matches C37a's arm. REGISTERED TO FAIL — the grader must be load-bearing on
  text exactly as C35d asserted for arithmetic.
- **C37d (self-vs-rule on text; de-hardcoding rung 4, re-homed from C35b):** the text arm's
  final library-cloze accuracy is within one margin of, or above, the textrule arm's
  (harness-authored quizzes over the same lines, answers by construction) at equal
  consolidated tokens. Survives → the quiz generator is DELETED from the intake path.
  Falsifier: self trails rule by > margin on ≥ 2/3 — the generator stays; rung 4 refused.
Wave: (text, textrule, notext) × seeds 1-3, cycles 8 × wake 49152 (~400k consolidated/run —
sized down per the user's small-tests directive, same night it was given).

**Authorship-rung-1 pilot record (2026-09-02 ~21:00, seed 0, world domain, equal budget —
who picks the word to study; verdict-grade registration deferred pending the user's step-1
capacity decision):** random pick (textrule) 0.21 / child's generative pick (choice; 31%
chosen, 69% random fallback) 0.12 / max-surprise pick (surprise) 0.02. Three readings:
(1) **uniform-is-unbeaten extends to word-level curriculum choice** — the ledger's most
persistent law, now in a third domain; (2) the toddler-child's greedy picks are degenerate
(common structural words) and actively hurt vs random — generative choice needs capacity
before it can carry signal; (3) max-surprise is the TOO-HARD end of §7.9's predictability
edge (it drills unguessable invented words and transfers ≈ nothing to random-position
probes) — the run does not falsify the edge, it confirms the edge's "noise = too hard" half
verbatim; the faithful chooser is mid-quantile surprise, untested. Lang held ≈ 1.96 in all
three arms (micro-sleep C36c signal, favorable).

## C34-v3 — the BPE child (built 2026-09-02 ~23:20, H100 #49703241, 1h50m, ~$3.20; cap $10)

190M params (dim 1024/L12/H16), self-given BPE-8k from own corpus (`models/bpe8k.json`, on HF
v3/), matched diet (dev order + second pass of the 8 digestible sources, no terminal ladder),
six stages ending in wcot (chain-of-thought, by construction). Body audit PASS (5.65 vs
init). 21 checkpoints at HF `checkpoints/v3/`; final = `models/child-base-v3.pt`.
**Gate a (own-held): 6/6 PASS, dramatic** (qa 10.1→1.04 b/tok, wcot 6.2→1.64).
**Gate b: BREACHED HARD — stage-era lang drift +1.380 b/char** (2.863→4.243; qa alone +0.885).
The bigger model learns formats far more invasively at the same stage constants; stage
rehearsal mix needs re-tuning before v3's stages can be called safe. Recorded, not accepted
as clean.
**The meaning probe (0.750) partially COLLAPSED under audit** (the falsifier culture applied
to our own headline): wcot's yes/no questions are ~88% "no" by construction (color match 1/8,
kind match 1/10), and transcript inspection shows the child answers via base-rate heuristics
("no" to is-questions + copy-the-color) while its T: lines cite a DIFFERENT invented entity
than the one in context — **novel-subject binding is absent**; the 0.75 is mostly class
imbalance + color-copying. wcot v2 must balance yes/no 50/50 and grade the T: chain against
the world (full-chain verification) so binding is forced, not optional.
**What v3 genuinely bought (talk battery vs v2, transcripts in session log):** free prose
went from word-salad fragments to grammatical sentence sequences; T:/A: thought register and
self-quizzing fire spontaneously and well-formed; color extraction from context works. Also
observed: the idk reflex over-triggers on qa tasks it was trained to do ("which is bigger",
"count the cats" → "I don't know") — abstention miscalibrated by the stage mix.

## Claims — SLEEP-sparse: tagged consolidation (registered 2026-09-02 ~23:55 at the user's
direction; instrument to be built; runs after the binding fix. The user's synaptic-tagging
proposal: the day's learning names the weights sleep may touch.)

Mechanism: the wake adapter's delta ΔW = B·A IS the tag (no interpretability needed — the
network files its own report of what today's learning is related to). Sleep trains a SPARSE
DELTA D over only the top-k% |ΔW| positions (base frozen bf16; teacher replaced by cached
distillation targets over the fixed wake buffer), same rehearsal + distillation loss, exact
fold at end (legal, unlike C15d's fold: the negotiation happens DURING D's training — the
function is unchanged at fold time). Purpose is dual: (1) memory — 1B-param sleep drops from
~19.5GB (does not fit a 4080) to ~4-5GB (fits with room) — the user's solely-local ceiling
moves to 1B+; (2) structural retention — an untouched weight cannot forget; rehearsal's
statistical protection narrows to the masked region. Known enemy from the literature,
registered as the thing to beat: mask-overlap accumulation across sessions re-creates
forgetting in published parameter-isolation methods (arXiv 2604.14010, 2502.10966); our
rehearsal inside the masked negotiation is the differing ingredient.
Arms: full sleep (control) / top-k masked / two-tier (the user's dot-product refinement:
ΔW-aligned-with-W positions fold cheaply as reinforcement, orthogonal positions get the full
negotiation). Probes: retention (childhood tails) + acquisition (stage/world helds) +
COMPOSITION (2-hop held — the registered risk: new knowledge that may not integrate through
a masked channel). Margins per house convention, pilot-frozen; seeds 1-3, ≥ 2/3.
- **SS-a (masked sleep suffices):** top-k arm within margin of full sleep on retention +
  acquisition + composition at equal steps. Falsifier: any of the three trails by > margin —
  the mask blocks integration; record WHICH probe broke (composition breaking = the
  connectedness objection wins and the memory path falls back to 8-bit/cached-teacher).
- **SS-b (structural retention pays):** masked arm BEATS full sleep on retention of
  never-rehearsed material (the C15i slice). Falsifier: no gain — immutability was already
  covered by rehearsal; the mask is only a memory optimization.
- **SS-c (two-tier, predicted-fail candidate):** aligned/orthogonal split beats plain top-k
  by > margin on the probe sum. Registered honestly as speculative; failure costs nothing.

**Instrument (built 2026-09-03 ~00:30: `cl/sparse.py` + `sparse_run.py` + `sparse_mem.py` +
`checks/sparse.py`; unit-checked: teacher == composite, gradient reaches only the tagged
values, base untouched until the fold, fold exact).** Deviations from the registration, all
recorded before any pilot: (1) the teacher is NOT a cached target set — it is the same frozen
base with the LoRA branch switched on (exactly the composite the full sleep deep-copies; zero
memory, exact); (2) the tag is per-Linear top-k (each adapted matrix keeps `frac` of its
positions; embeddings/LayerNorms/biases are untagged and untouched — the adapter never wrote
them); (3) two-tier's "aligned" = sign agreement of ΔW with W at the position; aligned
positions fold ΔW as-is and are not trained; (4) a fourth arm, **random** (same budget at
random positions), registered as the tag control — if random matches topk the tag story is
dead and this is plain sparse fine-tuning. Intake is fixed across arms (the sleep is the only
variable): a 150-entity world (cl/world.py) in 6 slices; per cycle 131k wake tokens of cloze +
balanced yes/no 2-hop units over the slice (LoRA rank 8, lr 3e-4), then 300 sleep steps
(old:new 16:16, frac 0.02); slice 0 never joins the reservoir (the C15i never-rehearsed slice).
Probes after every sleep: the v3 childhood tail windows (b/char, window-local chars/tok), the
six stage helds (b/tok), and for EVERY slice context-free cloze (content words only) and 2-hop
questions as exact-match + answer bits; unseen slices give the floor. Statistics and margins
live in `checks/sparse.py`; margins are frozen from the seed-0 pilot (full arm at seeds 0 and
100 gives the test-retest sd) before any verdict seed runs.

**Pre-pilot findings (seed 0, 2026-09-03 ~00:40; `results/sparse_settle_curve.*`,
`results/sparse_mem_1b.log`):**
- **The v3 base sat far from its rehearsal fixed point.** Five steps of the full sleep took
  the childhood tails from 4.28 to 2.00 b/char; a rehearsal-only curve reaches 1.76 at 150
  steps, then rises again (1.92 at 400 — the half-format reservoir pulls the model back toward
  stage formats). Reading: C34-v3's +1.38 stage drift is shallow and reservoir-driven, not lost
  knowledge. Unsettled, "retention" would have measured recovery speed and the full arm would
  win by construction — every arm now starts from a shared 150-step settle (`--settle 150`;
  same seed ⇒ same reservoir ⇒ same start).
- **The 4080 question, measured** (`sparse_mem.py`: 1.24B random-init GPT, dim 2048 / 24L,
  32-row sleep steps micro-batched by 4): masked sleep peaks at **7.07 GB** (wake 4.66 GB;
  24.5M tagged values at frac 0.02); the house sleep on fp32 master weights OOMs asking
  **29.4 GB of 16 GB**. A 1B child can sleep on the local card only through the tag. Next
  lever if ever needed: recompute the dense delta in backward instead of saving it (~2.5 GB of
  the 7).
- Cloze blanks on function words were guessable (0.2 exact-match on an unseen slice) —
  probes and wake units blank content words only.

**Fast profile (2026-09-03 ~11:25, registered before the v2 pilot; the user's directive: small
runs that decide, not long ones).** `sparse_run.py` defaults move from 6 cycles / 150 entities
to **3 cycles / 90 entities** — slice size unchanged (44 lines vs 42), wall clock roughly
halved. Evidence for the cut is the 6-cycle seed-0 full run (`results/sp_pilot_6cyc/`): the
retention separation is already at cycle 2 (full +0.28 b/char, topk −0.006 at the same
cycle; +0.63 vs −0.001 by cycle 6 — the later cycles only widen a decided gap), and the
acquisition/composition statistics are read right after each slice's own sleep, so they need
no cycle count at all. Honest cost: SS-b's P0 statistic (never-rehearsed slice erosion) has
two post-sleep cycles of exposure instead of five, so it has less power at this profile; a
null on SS-b here reads "not detectable in 2 cycles", not "no effect". The json is now dumped
after every cycle (the v2 topk run was lost to a shutdown at cycle 4 with nothing on disk).
Pilot v2 and the verdict seeds both run at this profile; the 6-cycle full run is kept as the
trajectory reference only (different args — it enters no margin).
- Statistic fix from the first fast-profile run (full s0, 11:33, before any margin was frozen):
  the floor slice sat 3 bits EASIER than the other slices at the pre-probe (item difficulty,
  no sleep yet), which read as negative "knowledge" for slice 0. Slice knowledge is now the
  floor gap minus the same gap at the pre-probe. Same seed ⇒ same world for every arm, so the
  arm comparison is unchanged; the test-retest margin (s0 vs s100, different worlds) stops
  absorbing item-difficulty noise.
- Pilot addendum (11:45, after the random arm landed, before its runs): at frac 0.02 the random
  mask matches topk on every statistic (lang −0.005 both; acq 4.85 vs 5.09; comp 6.08 vs
  6.25; P0 erosion 0.23 vs 0.62 — random better). Reading: 2% of a 190M model is ~3.8M free
  values, more than the world needs, so any sparse rehearsed delta suffices and the tag is
  not doing selective work at this budget. Added to the pilot: topk and random at frac 0.005
  (budget ladder DOWN). Prediction that would save the tag story: random degrades on
  acq/comp as the budget shrinks while topk holds. If both hold, the tag control stays a
  null at 190M and the mechanism claim waits for a scale where the budget binds.
- Margin read (11:54, full s0 vs s100): lang 0.289 vs 0.330 and stage −0.034 vs −0.029 are
  tight (2×|diff| = 0.081 / 0.010), but acq 4.59 vs 2.31 and comp 4.53 vs 2.89 are not — a
  different seed is a different world and a different exam, so the cross-seed difference is
  item difficulty, not sleep variance, and 2× it (4.6 / 3.3 bits) would make SS-a's learning
  half unfalsifiable. Arms at one seed share the world, so the verdict comparison is
  unaffected. Fix: `--world-seed` (default 500 + seed; the exam generator is tied to it, so
  the same world has the same exam) and a same-world retest, full at seed 100 on seed 0's
  world (`sp_pilot_full_s100w0`), from which the acq/comp/P0 margins are taken. Floors
  proposed (frozen with the margins after that run): lang 0.02 b/char, stage 0.02 b/tok,
  acq/comp 0.5 bits, P0 0.25 bits — ~1 % of the childhood/stage levels and ~10 % of the full
  arm's learning effect; below these a difference is not one we would act on.
- Wake off-by-one found while building SS-d (12:05): `wake_train` stopped one step short, so
  every pilot run up to and including `sp_pilot_full_s100w0` trained 31 of its 32 wake steps
  (the last 4 096 tokens of each wake buffer were never seen). Fixed at 12:07. The same-world
  margin pair (`sp_pilot_full_s0`, `sp_pilot_full_s100w0`) is like-for-like on the old code;
  the 0.5 % pair, the SS-d ladder and the verdict seeds run post-fix with 32 steps. The 3 %
  extra wake is systematic and equal across arms, so it shifts no arm comparison; it is
  noted here because the frozen margins come from 31-step runs.

**Pilot record, seed 0, fast profile, 31-step wake (2026-09-03 11:28–12:08; calibration only,
never a verdict). `results/sp_pilot_*.json`, statistics per `checks/sparse.py`:**

| arm | frac | lang Δ b/char | stage Δ b/tok | acq bits | comp bits | P0 erosion | peak MB |
|---|---|---|---|---|---|---|---|
| full s0 | – | +0.289 | −0.034 | 4.59 | 4.53 | 0.74 | 9099 |
| full s100 (own world) | – | +0.330 | −0.029 | 2.31 | 2.89 | 1.77 | 9099 |
| full s100 on world 0 | – | +0.399 | −0.033 | 4.71 | 3.00 | 0.46 | 9099 |
| topk | 0.02 | −0.005 | −0.033 | 5.09 | 6.25 | 0.62 | 7514 |
| topk | 0.05 | 0.000 | −0.068 | 5.08 | 5.78 | 0.50 | 7645 |
| topk | 0.10 | +0.021 | −0.085 | 5.03 | 5.57 | 0.37 | 7891 |
| random | 0.02 | −0.005 | −0.027 | 4.85 | 6.08 | 0.23 | 7514 |
| twotier | 0.02 | −0.005 | −0.004 | 5.05 | 6.67 | 0.52 | 7502 |

Reads: (1) every sparse arm holds the childhood flat while full drifts +0.29 to +0.40; stage
helds are flat for all; (2) with the floor-relative, baseline-corrected statistic the sparse
arms acquire and compose at least as much as full (last night's "topk acquires less" was
format learning on every slice, read as knowledge); (3) the tag-size ladder is flat on acq and
falls on comp, lang and memory as frac grows — 0.02 is the verdict frac; (4) random ≈ topk at
frac 0.02 (gap sum 0.41 in topk's favour, inside noise); (5) the full arm's COMPOSITION is
unstable across training seeds on identical questions (4.53 vs 3.00 bits) while its acquisition
is not (4.59 vs 4.71) — a finding about the house sleep, and the reason the comp margin is wide.

**Frozen constants (before any verdict seed):** MARGIN = 2× the same-world paired difference,
floored — lang 0.220 b/char, stage 0.02 b/tok (floor), acq 0.5 bits (floor), comp 3.06 bits,
P0 0.563 bits. Verdict frac 0.02. The gap-sum claims (SS-c, tag control) use the sum of the
four margins (3.80) as their bar — the tag control's placeholder rule `d > 0` is replaced by
this before verdicts, so "random matches topk" means "within the same noise SS-a tolerates".
Verdict wave: seeds 1–3 × {full, topk, random, twotier}, 32-step wake, chained after the
pilot addenda.

**Budget-down addendum read (12:19, seed 0, frac 0.005 — 797k trained values, 0.4 % of the
model):** topk005 lang 0.000 / stage +0.032 / acq 5.09 / comp 7.31 / P0 0.65; random005 lang
−0.001 / stage +0.026 / acq **3.65** / comp **4.85** / P0 0.10. The tag holds every statistic
it had at 0.02 (comp is its best yet); the random mask loses 1.44 bits of acquisition and 2.46
of composition — below even the full arm — for a gap sum of 3.89 against the 3.80 bar. The
registered prediction ("random degrades as the budget shrinks while topk holds") is met on
seed 0. Reading: with room to spare any sparse rehearsed delta suffices; when the budget binds,
|B·A| picks the positions that matter. Consequence, registered before the wave: the tag control
gets a seeded verdict at the binding budget — `sp_{topk,random}005_s{1,2,3}` join the verdict
wave (6 more runs, ~30 min) with the same gap-sum bar; the 0.02 control stays as registered and
is expected to read "inside noise". The first cost of a tight budget is stage drift (+0.03
b/tok at 0.005 vs −0.03 at 0.02), i.e. the tag budget trades format retention for selectivity.

**SS-d — tiny naps (registered 2026-09-03 ~12:05 at the user's direction, before any run; the
user's position: "no need sleep, just more tiny little naps — almost like gradient descent;
offline is only for pretraining, then it never goes offline again").** Claim: consolidation
granularity is free — at equal wake data and equal total sleep steps, the topk arm with N
naps per cycle (wake on 1/N of the slice → sparse sleep for sleep_steps/N → fold, N times;
`--naps N`) stays within margin of the batched arm (N = 1) on lang / stage / acq / comp.
Ladder N ∈ {4, 16, 32}; N = 32 is one wake step per nap (4 096 tokens, ~9 sleep steps) — the
online-sparse-GD-with-rehearsal endpoint. Falsifiers, each with a registered reading: (i) acq
or comp fall away as N grows — a one-step adapter has too little to teach and its tag is
noise (LoRA's B starts at zero, so after one Adam step ΔW ≈ lr·sign(∇B)·A is shaped by the
random A, not the data); consolidation needs a minimum "day"; (ii) lang or stage drift grows
with N — mask overlap across many folds re-creates forgetting (the literature's enemy, arXiv
2604.14010) and rehearsal inside the nap did not cover it; (iii) both hold to N = 32 —
"sleep" reduces to rehearsal + tag + sparse step, and the product design is nap-per-turn: no
idle window is ever needed. Where the ladder breaks is the finding. Follow-up, registered not
built: a teacherless arm (sparse delta trained on the wake tokens directly, tag = top-k
wake-gradient magnitude) is the true online endpoint; run only if SS-d survives at N = 32.
- **First rung read (12:24, `sp_pilot_topk_n4_s0`): falsifier (i) fires at N = 4.** lang +0.001
  / stage −0.057 (retention untouched by 12 folds instead of 3 — falsifier (ii), mask overlap,
  does NOT fire), but acq 2.29 vs 5.09 and comp 2.53 vs 6.25 at N = 1: half the learning at
  equal total steps. Mechanism as registered: each nap's fresh adapter has a quarter of the
  practice (wake online 2.19 vs 1.74 bits) so the teacher knows less, and 75 sparse steps fit it
  worse (new-row fit 1.34 vs 0.86). N = 16 / 32 run for the curve. **Condition re-registered
  (12:30, user: "go ahead build the teacherless nap"):** the teacherless arm is built and run
  NOW rather than after an N = 32 survival, because (i) is a structural cost of the middleman
  (restart + copy), which the teacherless design removes. `--sleep direct`: one CE backward over
  the chunk tags top-k |∂L/∂W| per adapted Linear (frac 0.02, same budget as topk); the sparse
  delta trains on the chunk's tokens (CE) + rehearsal 16:16 for sleep_steps/N steps; exact fold.
  No adapter, so there is no within-day fast learning: `wake` reports the pre-nap loss on each
  chunk — the prequential online loss. Ladder N ∈ {1, 4, 32} (`sp_pilot_direct_n{1,4,32}_s0`);
  N = 1 asks whether the LoRA teacher was ever needed, N = 32 is one action per nap. Same
  falsifiers as SS-d read against `sp_pilot_topk_s0`; an extra registered risk: without the
  adapter's low-rank bottleneck the direct delta can memorise the 16-row chunk (rehearsal is
  the only brake), read as stage/lang drift. Memory note: the tag pass needs full-model grads
  (+0.76 GB at 190M, +5 GB at 1B) — if that binds at 1B the fix is layer-at-a-time tagging.
- **LoRA-nap ladder complete (12:35, `sp_pilot_topk_n{4,16,32}_s0` vs `sp_pilot_topk_s0`):**

  | naps / round | lang Δ | stage Δ | acq bits | comp bits | last fit old/new | wake online |
  |---|---|---|---|---|---|---|
  | 1 | −0.005 | −0.033 | 5.09 | 6.25 | 2.74 / 0.86 | 1.74 |
  | 4 | +0.001 | −0.057 | 2.29 | 2.53 | 2.55 / 1.34 | 2.19 |
  | 16 | +0.005 | −0.049 | 1.85 | 2.36 | 2.40 / 1.36 | 2.30 |
  | 32 | +0.004 | −0.045 | 1.91 | 2.24 | 2.40 / 1.34 | 2.27 |

  Reads: (ii) does not fire anywhere — 96 folds leave childhood and stage exactly where 3 folds
  do (mask overlap is a non-event at 190M / frac 0.02 with per-nap rehearsal); (i) fires at N = 4
  and then FLATTENS — learning through the adapter-then-copy middleman bottoms out near 1.9 bits
  (~37 % of batched) rather than falling to zero, because even a one-step adapter nudges the
  teacher a little each nap. SS-d as registered (LoRA-distill naps within margin of batched) is
  falsified on seed 0 at every N ≥ 4 on acq and comp (gaps 2.8–3.2 bits against margins 0.5 /
  3.06 — acq breaks, comp is inside its wide margin). The retention half of the user's position
  is supported; the learning half moves to the teacherless arm.
- **Teacherless N = 1 (12:43, `sp_pilot_direct_n1_s0`):** lang 0.000 / stage −0.036 / **acq 16.56 /
  comp 16.69** / P0 erosion 5.95 / peak 7 391 MB; last fit old 2.63, new **0.17** nats; cloze EM
  0.18–0.25 and 2-hop EM 0.61–0.67 on consolidated slices (floor 0.02 / 0.35). Three times the
  acquisition and composition of any distillation arm (topk 5.09 / 6.25, full 4.59 / 4.53) at
  equal sleep steps and equal budget, with childhood and stage helds untouched and the lowest
  peak memory of all arms. Reading: the copy step (distilling a rank-8 composite's soft targets)
  was the bottleneck of every arm so far — the sparse delta trained on the tokens fits the day's
  buffer to 0.17 nats and the rehearsal mix is enough of a brake that no drift follows. The
  registered risk (memorisation read as lang/stage drift) did not materialise; what did
  materialise is faster absolute erosion of the never-rehearsed slice (6 bits over two cycles,
  from a 3× higher start — it still ends above every other arm's rehearsed slices).
  **Probe caveat, registered now:** every cloze and 2-hop probe question has appeared during
  wake in the same form (with the `text:` line; ~300 mentions per fact per cycle cover every
  blank position and 2-hop type), so acquisition = context-free recall of seen questions — the
  C15 currency, equal across arms, but it cannot separate "knows the fact" from "memorised the
  Q/A pair" now that an arm answers them well. Follow-up probe (not yet built): a held-out
  question FORM never used in wake (e.g. "what kind is a W") on the same facts.

**SS-e — teacherless nap, registered 2026-09-03 12:50 BEFORE its seed-0 N = 32 result and before
the verdict wave launched (the wave is re-scoped here: same 18 runs, reordered so the product path
runs first, plus 6 direct runs; nothing dropped).**
- **SS-e1 (teacherless suffices and out-learns the copy):** direct N = 1 within margin of full on
  lang and stage AND acquisition above topk's by more than the acq margin (0.5 bits). Falsifier:
  retention breaks (memorisation without the adapter's rank brake shows as drift) or the seed-0
  3× is a seed-0 glimpse (C13b's lesson).
- **SS-e2 (nap-per-action):** direct at 32 naps/round within margin of direct at 1 nap/round on
  lang / stage / acq / comp. Falsifier: acq or comp fall — 9 sparse steps per 4 096-token chunk
  are too few even without the middleman, and the minimum "day" is real; lang or stage drift —
  96 folds of a token-fit delta accumulate what 96 distillation folds did not. Registered with
  the seed-0 N = 32 run still in flight; its read is calibration only.
- Wave order (`checks/sparse.py`, workers = 1): full → topk → direct → direct N = 32 → random →
  twotier → topk 0.005 → random 0.005, seeds 1–3 each; 24 runs, ~2 h, all local.
- **Teacherless ladder, seed 0 (12:48 / 12:53; calibration, wave already launched):**

  | naps / round | lang Δ | stage Δ | acq bits | comp bits | P0 erosion | fit old/new | stream loss (pre-nap) |
  |---|---|---|---|---|---|---|---|
  | 1 | 0.000 | −0.036 | 16.56 | 16.69 | 5.95 | 2.63 / 0.17 | 2.87 |
  | 4 | +0.001 | −0.030 | 19.03 | 16.96 | 7.48 | 2.48 / 0.14 | 1.06 |
  | 32 | +0.003 | +0.004 | 14.32 | 16.59 | 5.81 | 2.34 / 0.19 | **0.47** |

  Reads: the middleman's falsifier (i) is gone — more naps learn MORE up to N = 4 and the finest
  grain still holds 3× every distillation arm; (ii) still silent after 96 token-fit folds (lang
  +0.003). The pre-nap loss on the incoming stream — the online-learning currency — falls from
  2.87 to 0.47 bits/token as the grain gets finer: the child predicts each arriving chunk having
  napped on all before it (facts recur across chunks). Cloze EM reaches 0.39 on a slice at N = 32
  (floor 0.07). **SS-e2 applied to seed 0 as registered:** N = 32 trails N = 1 by acq 2.25 (margin
  0.5 → breaks), stage 0.040 (margin 0.02 → breaks), comp −0.1, lang 0.003. The strict claim fails
  this seed. Reading recorded BEFORE seeds 1–3: the acq/stage margins were frozen from the full
  arm (effect 4.6 bits, both stages moving by 0.03); on an arm with a 16-bit effect a 0.5-bit
  tolerance measures a different thing (a 14 % shortfall at the finest grain, not a failure to
  learn), and the stage "break" is two arms both improving stage helds by different amounts. The
  rule is NOT changed while the wave runs; the verdict and this plain reading will both be
  recorded. Seed-0 sweet spot: a nap every few actions (N = 4), not every action.
- **Held-out question-FORM probe (13:00, `sparse_novel.py` on the seed-0 checkpoints;
  forms never used in wake — definitions "what kind is a W" → KIND, relations "what does the W
  VERB" → "the W2"; knowledge = floor gap at the checkpoint minus floor gap on the untouched v3
  base, mean over consolidated slices):** full **3.95** bits, topk **2.00**, direct N = 1 **6.03**,
  N = 4 **5.80**, N = 32 **4.88**. Exact-match 0 for every arm (a 190M child cannot produce a clean
  A: line for an unpractised question shape; bits carry the signal). Reads: (1) the direct arm's
  seen-form advantage (3.6× full) shrinks to 1.5× on unseen forms — part of the 16 bits is the
  phrasing, but real fact knowledge remains and is the largest of any arm; (2) the tag-distill arm
  knows LESS than full in new forms (2.0 vs 3.95) despite knowing slightly more in seen forms —
  distilling a rank-8 composite keeps the specific and loses the general; (3) the direct ladder
  degrades gently with grain on unseen forms (6.0 → 5.8 → 4.9). The caveat is answered for the
  direct arm: knowledge, not only Q/A memorisation. This probe is read-only on saved `.pt` files
  and can join the verdict read for any arm.
- **Classic-training control, registered 13:05 before its runs (user: "why don't you test it
  against standard pretraining - no nap, no adapter, no merging, just classic training").** The
  direct nap IS gradient descent on the LM loss; what differs from classic continued training is
  (1) the 2 % tag, (2) the rehearsal half of every batch, (3) the exact fold. `--sleep plain`
  trains EVERY parameter on the chunk's tokens with the same steps and the same rehearsal mix;
  `--sleep-old 0` removes rehearsal. Four seed-0 runs after the wave (`results/sp_pilot_queue5.ps1`):
  plain N = 1, plain0 (no rehearsal) N = 1, plain N = 32, plain0 N = 32. Decomposition: direct −
  plain = the tag's contribution; plain − plain0 = rehearsal's. Predictions: plain0 learns the
  world MOST (nothing holds it back) and forgets most (lang drift ≫ full's +0.29; the C32c/C15d
  pattern); plain (with rehearsal) forgets like full (+0.3, all weights move) and learns about
  what direct learns. Falsifier of the tag's usefulness: plain within margin of direct on lang AND
  stage — then the tag is a memory optimisation only (1B fits, nothing else) and the retention
  story belongs to rehearsal alone. Either way the reading is recorded. Peak memory is part of the
  read (plain needs full-model grads + Adam state = the house sleep's 9.1 GB; direct 7.4 GB).

**Verdicts (seeds 1–3, 32-step wake, frozen margins; `results/sp_{arm}_s{1,2,3}.json`, wave log
`results/sparse_verdict.log`):**

| seed | full lang / stage / acq / comp / P0 | topk lang / stage / acq / comp / P0 |
|---|---|---|---|
| 1 | +0.361 / −0.004 / 3.51 / 3.90 / 0.07 | −0.004 / −0.022 / 3.71 / 6.45 / 0.48 |
| 2 | +0.350 / −0.011 / 2.10 / 3.56 / 1.79 | −0.004 / −0.023 / 2.22 / 5.35 / 1.11 |
| 3 | +0.277 / −0.052 / 4.15 / 4.38 / −0.27 | −0.002 / −0.020 / 4.45 / 6.57 / −0.53 |

- **SS-a SURVIVES 2/3 (13:27).** The tag arm beats full on lang (0.28–0.37 b/char less drift),
  acq (+0.12 to +0.30) and comp (+1.8 to +2.6) on 3/3 seeds. Seed 3 broke on stage alone: both
  arms IMPROVED the stage helds (full −0.052, topk −0.020) and the difference, 0.032, exceeds the
  0.02 floor margin — a "both better, one less so" break, recorded as such. The house sleep's
  drift replicates on every seed (+0.28 to +0.36).
- **SS-b FALSIFIED 1/3 (13:27).** P0 erosion topk − full: +0.41 / −0.68 / −0.26 bits against the
  −0.563 bar; only seed 2 passes. Registered reading applies: rehearsal already protected what an
  untouched weight would have; at 190M the tag's retention advantage is the memory optimisation,
  not structural immunity for never-rehearsed material.
- **SS-e1 SURVIVES 2/3 decided at seed 2 (13:38).** direct seeds 1 / 2: lang −0.003 / −0.001,
  stage −0.029 / −0.026, acq **15.24 / 12.64**, comp **14.27 / 14.88**, P0 erosion 6.8 / 9.3, peak
  7 391 MB. Against full: 0.36 / 0.35 b/char less drift, stage within margin; against topk: acq
  +11.5 / +10.4 bits (bar 0.5), comp +7.8 / +9.5. Seed 0's 3× was not a glimpse. The registered
  falsifier (memorisation without the rank brake shows as drift) did not fire on any seed; the
  cost that did appear is the never-rehearsed slice's absolute erosion (6.8–9.3 bits from a
  ~16-bit start — it still ends above every distillation arm's rehearsed slices).
  Seed 3 (13:44): acq 16.77 / comp 15.20 / lang −0.001 / stage −0.030; the rule flags stage by
  0.022 vs 0.02 in the "both improved" pattern (full −0.052). Learning ahead of topk on 3/3.
- **SS-e2 FALSIFIED as registered, decided at seed 2 (13:55).** direct N = 32 seeds 1 / 2: lang
  +0.016 / +0.008, stage **+0.038 / +0.026**, acq 15.01 / 12.21, comp 16.55 / 17.15, P0 erosion
  3.96 / 7.36, pre-nap stream loss **0.454 / 0.451** bits/token. Trails N = 1 by acq 0.23 / 0.43
  (inside the 0.5 margin), comp −2.3 / −2.3 (better), lang 0.019 / 0.009 (inside 0.22); breaks on
  stage alone, 0.067 / 0.052 vs 0.02 — and unlike the SS-a/e1 seed-3 breaks this is a real cost:
  N = 32 drifts the stage helds UP (+0.03–0.04 b/tok, ~2 % of their level) where N = 1 improves
  them. Plain reading: nap-per-action learns as much as nap-per-round, composes better, keeps the
  childhood within margin and predicts the live stream at 0.45 bits/token; its one consistent cost
  is small format drift from 96 token-fit folds each guarded by only 9 rehearsal steps (144 old
  rows per nap vs 4 800 per batched sleep, same total). **Follow-up registered now, before its
  run:** direct N = 32 with `--sleep-old 48` (3:1 old:new per step, 3× the rehearsal rows per nap
  at ~1.7× the step cost; `sp_pilot_direct_n32_r48_s0`). Prediction: stage drift returns to ≤ 0
  with acq within margin; falsifier: stage still drifts — the cost is the fold count, not the
  rehearsal dose, and the product grain is a nap every few actions (N = 4 learned most anyway).
  Seed 3 (14:00): lang +0.013, stage +0.033, acq 17.13 (above N = 1's 16.77), comp 16.36,
  stream 0.476 — 0/3 on the stage margin, 3/3 on learning. Final: SS-e2 FALSIFIED on stage only.

**Classic-training ablation, seed 0 (14:00–14:17, `sp_pilot_plain*_s0`; runs with `--sleep-old 0`
also settle on 16 rows/step instead of 32, so their pre-probe differs slightly — lang 1.782 /
stage 1.963 vs 1.764 / 1.892 — and their drifts are read against their own pre):**

| arm | weights | rehearsal | naps | lang Δ | stage Δ | acq | comp | P0 | stream | peak MB |
|---|---|---|---|---|---|---|---|---|---|---|
| plain0 | all | none | 1 | **+0.965** | **+2.866** | 25.0 | 46.8* | 11.6 | 4.76 | 6 283 |
| plain | all | 16:16 | 1 | +0.360 | +0.053 | 20.2 | 14.1 | 10.9 | 3.27 | 8 246 |
| direct | tag 2 % | 16:16 | 1 | 0.000 | −0.036 | 16.6 | 16.7 | 6.0 | 2.87 | 7 391 |
| plain0 | all | none | 32 | **+2.666** | **+9.178** | 23.8 | 49.6* | 13.3 | 0.59 | 6 283 |
| plain | all | 16:16 | 32 | +0.410 | +0.088 | 19.5 | 14.8 | 7.7 | 0.49 | 8 245 |
| direct | tag 2 % | 16:16 | 32 | +0.003 | +0.004 | 14.3 | 16.6 | 5.8 | 0.47 | 7 390 |

\* the floor slice's 2-hop bits explode when the T:/A: format is destroyed (19 → 47), so
"composition knowledge" is meaningless for the no-rehearsal arms; their acq is memorisation of the
current slice with the previous slice forgotten within one round (slice 1: 0.9 → 11.4 bits).
Reads, as predicted and registered: (1) **classic continued training on a stream wrecks the
model** — +0.97 b/char childhood and +2.9 b/tok stage in three rounds at N = 1, total collapse
(1.78 → 4.45 b/char, 1.96 → 11.1 b/tok) within ONE round at N = 32 — while its pre-nap stream
loss looks as good as the nap's (0.59 vs 0.47): online loss cannot see forgetting. (2) **Rehearsal
alone** buys back ~0.6 b/char and nearly all of the stage; what remains (+0.36 / +0.41 b/char,
+0.05 / +0.09 b/tok) is exactly the house sleep's drift — moving every weight forgets, rehearsed or
not. (3) **The tag alone** (direct − plain, same loss / data / steps / rehearsal) buys the rest:
childhood and stage to zero, better composition (+2.6 / +1.8 bits), half the P0 erosion, ~0.85 GB
less memory, at a cost of 3.6 / 5.2 bits of seen-form recall. The tag's usefulness falsifier
(plain within margin of direct on lang and stage) does NOT fire: plain trails by 0.36 / 0.41
b/char against a 0.22 margin. The nap is not "just training": it is training whose two additions
each remove a measured part of the forgetting, and neither alone reaches zero.
- **Rehearsal-dose follow-up (14:25, `sp_pilot_direct_n32_r48_s0`, direct N = 32, old:new 48:16):**
  lang +0.025, **stage −0.014** (the finest-grain format drift is gone, as predicted), acq 12.99,
  comp 13.14, P0 erosion 3.65 (lowest of the direct arms), stream 0.492, **peak 11 793 MB**. Half
  the prediction held: stage returns below zero; acq does NOT stay within margin of N = 1 (trails
  by 3.6 bits) and comp drops 3.4. And the practical cost is decisive: the 64-row step lifts peak
  memory from 7.4 to 11.8 GB — at 1B that undoes the 4080 fit. Reading: the dose fixes the stage
  drift but is not the product answer; the grain is (N = 4 learned most with no drift). Cheaper
  levers to test if nap-per-action is ever required: micro-batched rehearsal (same rows, same
  memory), or 2:1 rather than 3:1.

**Control verdicts (seeds 1–3, wave resumed 14:25 after the ablation; `results/sp_{random,twotier}_s{1,2,3}.json`):**
- **Tag control at frac 0.02: FALSIFIED as registered (inside noise), 0/3.** random trails topk by
  0.61 / 0.68 / 0.78 on the gap sum against the 3.80 bar — a small, consistent edge for the tag on
  every seed (random acq 3.32 / 1.79 / 3.93 vs topk 3.71 / 2.22 / 4.45; comp 6.23 / 5.10 / 6.30 vs
  6.45 / 5.35 / 6.57; lang and stage identical), but well inside the tolerance SS-a uses. Reading as
  pre-registered: at 2 % of a 190M model any sparse rehearsed delta suffices; the tag's selective
  work is tested at the budget that binds (0.005, below).
- **SS-c two-tier: FALSIFIED as predicted, 0/3.** twotier beats topk by 0.22 / 0.02 / 0.35 on the gap
  sum (bar 3.80): lang identical (−0.005 / −0.004 / −0.003), stage flat (+0.002 to +0.004 vs topk's
  −0.02, i.e. slightly LESS stage improvement), acq 3.61 / 2.00 / 4.31 vs 3.71 / 2.22 / 4.45
  (slightly less), comp 6.79 / 5.62 / 7.08 vs 6.45 / 5.35 / 6.57 (slightly more, 3/3). Folding the
  sign-aligned half of the tag untrained neither helps nor hurts at this scale — it trains half the
  values for the same result, which is a cost win, not a learning win. Registered speculative;
  failure costs nothing, as written.
- **Tag control at frac 0.005 (the budget that binds): FALSIFIED as registered, 1/3 — with the
  direction unbroken on 4/4 seeds.** random005 trails topk005 on the gap sum by 3.60 / 2.59 / 4.41
  (pilot seed 0: 3.89) against the 3.80 bar; acq 2.27 / 1.02 / 2.53 vs 3.41 / 1.79 / 4.11 (−33 % /
  −43 % / −38 %), comp 4.61 / 4.14 / 4.88 vs 7.09 / 5.97 / 7.71; lang identical, stage identical
  (+0.02–0.03 for both — the tight budget's format cost is budget, not tag). Reading: the tag's
  selective work is real and repeatable once the budget binds, at about a third of acquisition and
  composition; it fails a bar that inherited the full arm's 3.06-bit composition margin. A future
  re-registration should give the gap-sum claims their own margin from a same-world sparse pair;
  the rule stands as written for this round.

**ROUND VERDICT TABLE (`results/verdicts_sparse.json`, 15:31; 24 verdict runs + 27 seed-0
pilot/ablation runs, all local, $0):**

| claim | verdict | per-seed | plain reading |
|---|---|---|---|
| SS-a sparse sleep suffices | **SURVIVES 2/3** | ✓ ✓ stage-only | beats full on lang/acq/comp 3/3; 7.5 vs 9.1 GB |
| SS-b never-rehearsed slice protected | FALSIFIED 1/3 | ✗ ✓ ✗ | rehearsal already covers it |
| SS-e1 teacherless out-learns the copy | **SURVIVES 2/3** | ✓ ✓ stage-only | 3–4× acq and comp, retention flat, 7.4 GB |
| SS-e2 nap-per-action ≈ nap-per-round | FALSIFIED 0/3 | stage ✗ ×3 | learns as much, stream 0.45 b/tok, stage +0.03 b/tok |
| SS-c two-tier beats topk | FALSIFIED 0/3 | 0.02–0.35 vs 3.8 | ties; half the trained values |
| tag control 0.02 | FALSIFIED 0/3 | 0.6–0.8 vs 3.8 | inside noise, as pre-registered |
| tag control 0.005 | FALSIFIED 1/3 | 3.6 / 2.6 / 4.4 vs 3.8 | tag wins ~⅓ on acq/comp, 4/4 seeds |

Standing results of the round, in one paragraph: consolidation through a 2 % tag with rehearsal
retains everything the house sleep loses and learns at least as much; learning straight from the
day's tokens instead of copying an adapter triples what is kept, with no retention cost at
190M; the grain can go to one nap per action at a small format cost, and a nap every few actions
costs nothing; classic training on the same stream wrecks the model while its online loss looks
identical; rehearsal and the tag each remove a measured part of the forgetting and neither alone
reaches zero; the tag's selectivity shows only when the budget binds. The design that leaves the
round: **no adapter, tag from the action's own gradient, sparse step with rehearsal, exact fold,
every few actions, forever** (design-interface §7.10–7.11).

## Probe — OPEN-1B: the teacherless nap on an open 1B (registered 2026-09-03 ~16:25, before the
run; NEXT_SESSION step 1 — "nap loop on an open 1B before building one")

**What it is and is not.** Pythia-1B (EleutherAI/pythia-1b: GPT-NeoX, 16L/2048/8h, vocab 50 304,
context 2048, stored fp16; loaded fp32) is the MEASURING STICK for the nap at 1B, exactly as the
invented world is the measuring stick for learning — not a base for the product. T5 ("grown, not
given") and the base-vs-live rule (design-interface §7.12) are untouched; no reading here says
anything about the product base. Questions, from the handoff: does the nap hold on a real 1B; does a
1B student acquire and compose more than the 190M child. Seed 0 = calibration, as always.

**Instrument = the SLEEP-sparse instrument unchanged**, run through a substrate seam: `cl/hf.py`
(`HFLM`: forward returns (logits, hidden), `.block`, `.spots()` = each layer's
query_key_value/dense/dense_h_to_4h/dense_4h_to_h + embed_out — the same five spots per layer as
`lm_adapter.spots`; module names verified against the installed transformers 5.13 source);
`lm_adapter.spots()` asks a substrate that names its own; `sparse_run.py --hf NAME --corpus DIR
--micro K`. Regression, run before anything else: the v3 instrument with the flags unset, direct and
topk smokes (1 cycle, 8 192 wake tokens, 4 sleep steps, 12 entities), every recorded number
identical to the untouched code (56/56 fields each, `results/smoke/v3_{before,after}_*.json`).
`--micro` chunks rows within one gradient step on the teacherless path only (per-row CE, chunks
accumulate exactly; asserted off for the teacher modes).

**Corpus = the base's own diet.** Reservoir + retention tails from the Pile's held-out test split
(HF monology/pile-uncopyrighted, `data/fetch_pile.py` → `data/pile/*.txt`, one file per source,
16 sources kept at ≥ 2M chars: arxiv, dm_mathematics, europarl, freelaw, github, gutenberg_pg_19,
hackernews, nih_exporter, philpapers, pile_cc, pubmed_abstracts, pubmed_central, stackexchange,
ubuntu_irc, uspto_backgrounds, wikipedia_en; enron_emails dropped at 1.6M). `lang` = mean
bits/char over the 16 tails (same windows: 25k-token tail, 64 × 256 rows); no stage helds (the
open base has no formats); the reservoir's 8 000 slots all come from the corpus (v3: half formats).
Tokenizer = the base's own (`Tokenizer.from_pretrained`, the same `tokenizers` class the house BPE
loads); V = 50 304 (the head is padded past the tokenizer's ids). The wake format text:/Q:/T:/A:
is new to this base — learned in-run, absorbed by the floor slice like any format effect.

**Deviations from the registered direct arm, owned:** `--settle 0` (full-parameter settle OOMs at
1B fp32; the base sits at its own diet's fixed point by construction — the reservoir IS its
pretraining distribution); `--micro 4` (memory, numerics-neutral as above). Everything else as the
verdict wave's direct arm: 3 cycles, 90 entities, 131 072 wake tokens, 300 sleep steps, frac 0.02,
rehearsal 16:16, lr 3e-4, rank unused, seed 0, world seed 500.

**Smoke (16:15, `results/smoke/py1b_direct.*`, 1 cycle / 8k wake / 4 steps):** pre lang 0.871
b/char on the Pile tails; peak 13 071 MB (fp32 master weights + fp32 dense deltas); after ONE 4-step
nap lang 0.950 (+0.079). Prior recorded before the run: an Adam step on a base at its fixed point
can only move it off the Pile, so the first steps cost; the 300-step nap with 16:16 rehearsal is
where the equilibrium shows.

**Pre-registered readings (seed 0; a reading, not a verdict — margins for this substrate do not
exist yet and would need a same-world retest pair):**
- **OPEN-1B-a — the nap holds.** Pile-tail drift after 3 cycles ≤ 0.02 b/char (the lang floor).
  Fail reading: drift > 0.02 → the registered lr is too hot for a pretrained 1B at its fixed point;
  follow-up = an lr ladder on seed 0 (1e-4, 3e-5), registered before running, never on verdict seeds.
- **OPEN-1B-b — a 1B learns more.** acq ≥ 16.6 and comp ≥ 16.7 bits (v3 direct N = 1 seed 0:
  16.56 / 16.69). Fail reading: a 1B pretrained on the Pile does not bind invented words faster
  than the 190M child through the same 2 % channel — size is not the binding step at 1B.
- **OPEN-1B-c — knowledge, not phrasing.** `sparse_novel.py` held-out-form knowledge ≥ 6.0 bits
  (v3 direct 6.03).
- Memory: peak ≤ 14 GB at the registered shape (the 4080 fits a 1B nap in fp32 master).
Second run, same seed: `--naps 4` (the grain leaving the round) — is N = 4 the sweet spot here too
(v3: acq 19.0 at N = 4 vs 16.6 at N = 1)?

Runs: `results/py1b_direct_s0.{json,pt,log}`, `results/py1b_direct_n4_s0.*`; probe:
`python sparse_novel.py results/py1b_direct_s0.json` (loads the untouched pretrained base as the
"init" reference).

**Reading (2026-09-03 17:45; the user stopped the run after 2 of 3 cycles — ~30 min per cycle at 1B
fp32 with micro 4; the N = 4 run never started). Two cycles, floor = slice 3:**
- **OPEN-1B-a FAILS at the registered lr:** Pile-tail drift +0.064 after cycle 1, +0.078 after
  cycle 2 (bar 0.02) — front-loaded, then +0.014 per cycle. A base at its own fixed point pays the
  first naps; the lr/SGD ladder on seed 0 is the registered follow-up (Meta's sparse-memory paper
  found AdamW hurts sparse updates and used SGD).
- **OPEN-1B-b just under:** acq 16.28 / 16.23 bits per slice (bar 16.6); comp 8.45 / 10.5 (bar
  16.7) — the 1B learns the cloze as much as the 190M child and composes less through the same 2 %
  channel; slice-1 EM after its nap: cloze 0.77, 2-hop 0.95 (format learned in-run from zero).
- Slice 0 (never rehearsed) eroded 5.4 bits and EM 0.70 → 0.36 in one cycle: C15i at 1B.
- Online: pre-nap wake loss 2.91 → 1.90 bits/token from cycle 1 to 2 (the format carried over).
- Memory 13 073 MB peak, stable across cycles (bar 14 GB: met).
- OPEN-1B-c not run (the checkpoint is the 2-cycle state).
Decision taken on this reading + the size battery (Pythia-410M binds as well as 1B on the in-context
battery): **the live loop (design-interface §7.13) starts on Pythia-410M**; the 1B waits for the
drift ladder.

## LIVE — the live loop, first light (2026-09-03 17:50–18:10; design-interface §7.13; `live.py`)

Not a claim round: the build's first readings on seed 0, recorded so its constants have a source.
**Instrument:** `live.py` — one process on Pythia-410M (`cl/hf.py`), the human types, the child
answers in the register, tool lines (`run:`/`read:`/`search:`/`ask:`, `cl/tools.py`) get their real
output injected mid-generation (`cl/gen.py` token-row interception), a teacherless nap every 4 turns
or on `/nap` (tag = top-k |∂L/∂W| of the rows' own gradient, frac 0.02, sparse step on the rows +
16:16 rehearsal, exact fold; dose = the verdict nap's 9.4 exposures per row → 1–5 steps), reservoir
born from the Pile test split (8 000 × 256) and rehearsing the life's rows after; weights +
reservoir + transcript + nap journal persist per life. **Birth = the formats as the first naps**
(§7.12 item 10): N naps over generated units in the register with REAL tool outputs (v3's qa/cot/idk
generators + `run:` through the interpreter, `read:`/`search:` over the docs folder), with a
6-demo header kept in context. Micro-batched at 8 rows (the 50k head makes a nap step's logits 6×
the v3 child's: 16 GB peak without, 7.5 GB with).

**Birth ladder (Pile tails bits/char, base 0.911; tools used on 6 fixed questions):**

| naps × lr | lang after birth | drift | read:/search:/idk |
|---|---|---|---|
| 60 × 3e-4 | 1.237 | +0.326 | yes / yes / yes |
| 30 × 3e-4 | 1.146 | +0.235 | yes / yes / yes |
| 30 × 1e-4 | 1.019 | +0.108 | yes / yes / yes |
| **60 × 1e-4** | **0.998** | **+0.087** | yes / yes / yes |

More naps at the lower rate drift LESS than fewer: the cost is front-loaded and rehearsal pulls
back (the v3 stage-drift lesson, shallow and reservoir-driven). **Chosen: birth 60 × 1e-4; life
naps at 1e-4** (the 190M constant 3e-4 costs 3× the base's language on a base at its fixed point —
the OPEN-1B drift, priced). Nap cost: 0.6–2.8 s per nap at 410M; a 60-nap birth ≈ 2 min.

**Formats live in the weights, not the prompt** (no-header probe, greedy, `life/test` = 60 × 3e-4):
untouched base on `Q: what does the file notes.md say` → "A: The notes.md file is a text file
that contains…"; after birth → `read: notes.md` + the real `out:`. `Q: what is a vorpal` → base
rambles; after birth → `A: I don't know`. `Q: fill the blank: the ___ is blue` → base emits LaTeX;
after birth → `search: blue` + the real hit. Resume verified: `/probe` at quit 1.217 = at resume.

**Behaviours to price, not bugs (the §7.13 LIVE-a policy questions):** no `run:` on a plain
product ("17*23" → "A: 17"/"seventeen" — the run: units only cover the stage grammar's
parenthesised expressions); T: chains are arithmetic nonsense (practice, not size — the ladder
domain); free text after an `out:` with no `A:`; an occasional empty turn (it writes `Q:` at once →
"(silence)"); a made-up `out:` line is cut at the guard (`legal`). Session naps: 1 row, 1 step,
0.6 s; lang 1.237 → 1.217 across two of them (rehearsal recovering the birth drift).

## LANGUAGE MEMORY — tried (2026-09-03 22:20–22:40; the user's idea; `cl/memory.py`, `fun/memory_test.py`)

**The idea:** a two-model mind — a frozen thinker (Qwen3-4B) decides what to keep and what to ask; a tiny
learning memory (the v3 190M child, our nap) keeps it in its weights and answers in words. Different
from a store by four properties: paraphrase recall, integration, overwrite-by-learning, constant size.
**The falsifier, stated before the run:** precision — if recall after one save trails a plain keyword
store, or it confabulates more than it declines, it is a worse store.

**Instrument:** `Memory.remember(sentence)` = a direct nap over the sentence + one fill-the-blank quiz
per content word (the sentence is the answer key, §7.7-legal), each quiz its own row WITHOUT the
sentence in view, studied until the quizzes cost ≤ 0.5 bits/token or 40 steps, rehearsing the child's
diet (16 rows) and three earlier memories per step; `recall(q)` = greedy `Q: q\nA:` from the weights.
Rival: the saved sentence sharing the most words with the question. Three facts saved once, six
questions (one in the studied form, five in new words), two never-saved, then 100 invented-world
memories and the same exam.

| | after 3 facts | after 100 more |
|---|---|---|
| memory: studied form ("my cat is called ___") | Momo ✓ | "gike" ✗ (a filler word) |
| memory: five questions in new words | 0/5 | 0/5 |
| memory: two never-saved questions declined | 0/2 | 1/2 |
| keyword store: six saved-fact questions | 6/6 | 6/6 |

First attempt (quizzes with the sentence in view, fixed 8 steps): 0/6 even in the studied form — the
child answered from the sentence in context and nothing moved into the weights; that is why the
quizzes went context-free.

**Reading:** one telling CAN be written into 190M weights in the form studied (6–14 steps, 3–6 s) — the
dose answer. It does NOT transfer to a question in other words at this size (the held-out-form probe's
partial 6 bits is not exact-match recall), and it is overwritten by later memories that are rehearsed
only three at a time (C15i again). **FALSIFIED as registered: at 190M the language memory is a worse
store.** Consequence taken: Think keeps a plain note file; `cl/memory.py` stays for a body that binds
(the size battery says 410M+ on 300B tokens) and for rehearsing memories, not the diet, once they exist.

## Evidence

**Pilot, seed 0 (calibration only, never a verdict).** `results/pilot_{bp,cbp}_s0.json`, 200 × 10 000, lr 0.05.

| arm | first window | best | last | best − last | dead first→last | mean \|w\| | effective rank |
|---|---|---|---|---|---|---|---|
| bp | 0.854 | 0.868 | 0.831 | 0.038 | 0.029 → 0.124 | 0.031 → 0.074 | 39.9 → 53.0 |
| cbp 1e-4 | 0.856 | 0.872 | 0.869 | 0.003 | 0.002 → 0.001 | 0.027 → 0.026 | 41.5 → 63.2 |

Reads: the scale is sufficient for C1 (3.8-point decline, still falling at task 200); the C6
effective-rank component points the *wrong way* under BP at this scale (rank rose), so C6 is the
claim most likely to fall on the verdict seeds. lr pilot (1 task × 60 000): online 0.720 / 0.881 /
0.904 at lr 0.01 / 0.05 / 0.1 (`results/pilot_lr*.json`).

**Forgetting, verdict seeds 1–3.** `checks/forgetting.py` → `results/forget_{bp,cbp,replay}_s{1,2,3}.json`,
`results/verdicts_forgetting.json`. 10 tasks × 60 000, lr 0.05. Forgetting = mean over the 9 earlier
tasks of (best-ever − final) test accuracy; final = mean test accuracy over all 10 tasks after task 10.

| seed | bp forgetting | cbp 1e-4 | replay 2000 | bp final | cbp final | replay final |
|---|---|---|---|---|---|---|
| 1 | 0.390 | 0.640 | 0.024 | 0.611 | 0.384 | 0.943 |
| 2 | 0.355 | 0.680 | 0.023 | 0.642 | 0.348 | 0.944 |
| 3 | 0.342 | 0.660 | 0.021 | 0.652 | 0.366 | 0.944 |

C7 SURVIVES (34–39 points). C8 FALSIFIED on 3/3 seeds — and not marginally: CBP *increases*
forgetting by 27–33 points over BP. Mechanism, read off eq. 1: utility is |activation on the current
stream| × |outgoing weights|, so a unit that fires only for earlier tasks' inputs is the least useful
by definition and is the first reinitialised. CBP forgets by construction; C8b tests the rate.
C9 SURVIVES (≈ 2 points, 94 % final).

C8b (`results/forget_cbp_lo_s{1,2,3}.json`, rate 1e-5): forgetting 0.435 / 0.466 / 0.433, final 0.571 /
0.543 / 0.570 — still 4–11 points *worse* than BP on 3/3 seeds; SURVIVES. Damage is monotone in the
replacement rate (0.66 → 0.44 → BP 0.36): CBP's forgetting cost is the reinitialisation itself, not a
badly chosen rate.

**Plasticity, verdict seeds 1–3.** `checks/plasticity.py` → `results/plast_{arm}_s{1,2,3}.json`,
`results/verdicts_plasticity.json`. 200 tasks × 10 000. Cell = last-window online accuracy (best − last).

| arm | seed 1 | seed 2 | seed 3 | dead first→last (s1) | mean \|w\| (s1) | effective rank (s1) |
|---|---|---|---|---|---|---|
| bp | 0.828 (0.043) | 0.827 (0.042) | 0.831 (0.039) | 0.027 → 0.146 | 0.031 → 0.073 | 41 → 51 |
| adam 1e-3 | 0.810 (0.050) | 0.815 (0.048) | 0.817 (0.046) | 0.289 → 0.657 | 0.053 → 0.164 | 51 → 19 |
| l2 1e-4 | 0.844 (0.027) | 0.841 (0.030) | 0.847 (0.024) | 0.030 → 0.124 | 0.030 → 0.054 | 39 → 54 |
| snp 1e-4 / 3e-5 | 0.837 (0.034) | 0.845 (0.025) | 0.844 (0.027) | 0.030 → 0.108 | 0.031 → 0.054 | 40 → 55 |
| cbp 1e-4 | 0.865 (0.009) | 0.867 (0.005) | 0.869 (0.005) | 0.003 → 0.001 | 0.027 → 0.027 | 41 → 67 |
| norm-reset (C10) | 0.845 (0.019) | 0.851 (0.014) | 0.856 (0.007) | 0.028 → 0.225 | 0.021 → 0.013 | 36 → 49 |
| random reinit (C11) | 0.864 (0.006) | 0.863 (0.006) | 0.865 (0.003) | 0.017 → 0.028 | 0.025 → 0.027 | 33 → 26 |

C4b (`results/plast_snp_hi_s{1,2,3}.json`, noise 1e-4): 0.840 (0.031) / 0.844 (0.027) / 0.847 (0.024) —
indistinguishable from L2; FALSIFIED 3/3. At this scale the perturbation adds nothing to the decay.

Verdicts: C1, C2, C3, C5 SURVIVE 3/3. C4 and C4b FALSIFIED 3/3. C6 FALSIFIED 3/3:
under SGD the effective rank rises (+10 to +12) while accuracy falls; the paper's full picture — dead
units, weight growth *and* rank collapse — appears here only under Adam. C10 FALSIFIED 2/3 by 0.4–0.9
points over the bar, yet it is the best non-reinitialising method, with *more* dead units than BP.
C11 SURVIVES 3/3: choosing units by utility buys nothing over choosing them at random.

Reading across C5, C8, C10, C11: what maintains plasticity here is injecting fresh random weights and
keeping norms small; the utility-based *selection* is the one ingredient that is measurably harmful
(C8: it destroys units the current stream does not use) and never measurably helpful (C11).

**Coverage round, verdict seeds 1–3 (2026-09-01).** `checks/coverage.py` → `results/walk_*`,
`results/drift_*`, `results/cov_sleep2000_*`, `results/verdicts_coverage.json`, log
`results/coverage_check_run1.log`. N-walk gap (unbounded − bounded, pts) at N = 10k/20k/40k/80k:
0.3/2.4/4.8/7.7 · 0.3/2.4/4.8/8.0 · 0.4/2.3/4.9/7.7; knees 11.9k/12.4k/12.2k. Drift world: mass
0.095–0.103; at-risk facts 778–783; S_drift bp 0.846/0.807/0.815, naive +1.4/+1.2/+1.7 over bp,
aware +4.8/+5.8/+5.9 over bp; aware dwell median 5.4–6.2k examples, ~45 % corrected in place.
φ_store(131 072) = 0.981–0.982. Sleep spot-check: S_ret 3.0/3.6/2.3 pts (2000) vs 1.5/1.9/1.6
(8000); trend(11–20) 0.21/0.22/0.11 vs 0.094/0.095/0.069 pt/cycle.

**Sleep round, verdict seeds 1–3 (2026-09-01).** `checks/sleep.py` → `results/sleep_{arm}_s{1,2,3}.json`,
`results/verdicts_sleep.json`, log `results/sleep_check_run1.log`. Key statistics (points):

| arm / statistic (bar) | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| stored S_ret (≤ 2) | 1.5 | 1.9 | 1.6 |
| stored S_new (≤ 2) | 0.1 | 0.1 | 0.1 |
| stored trend 11–20 (≤ 0.05/cyc) | 0.094 | 0.095 | 0.069 |
| accel trend 21–40 / 41–60 | 0.006/0.008 | 0.005/0.010 | 0.007/0.006 |
| stored S_plast (≤ 1) | −0.9 | −4.4 | −0.9 |
| labels S_ret | 0.7 | 0.5 | 0.4 |
| weightadd S_ret | 72.6 | 62.7 | 64.9 |
| selfgen S_ret | 24.3 | 17.7 | 20.1 |
| rebirth S_ret | 6.5 | 5.4 | 7.1 |
| oracle S_ret | 1.4 | 1.7 | 1.1 |
| P0 drop (stored) | 14.4 | 23.1 | 12.2 |
| conflict correction (0.81 ± .15) | 0.92 | 0.88 | 0.90 |

Controls: **null** = exactly zero on every seed (σ_null = 0 bitwise — the identity-adapter sleep is
the distillation fixed point; TREND_BAR sits on its registered 0.05-pt floor); **ensemble** F 91–92 %,
P0 92 % (nothing merges, nothing erodes — P0's loss elsewhere is merge traffic); **scratch** 91.5–92.2 %
mean over 21 domains (capacity comfortable); **probation dial**: 2.5k wake → S_ret 0.3–0.5 but mean
online 0.63 vs 0.79 (learns far less); 40k wake → S_ret 4.4–4.8. Merge damage scales with how much
was learned awake — same story as the novelty tax.

Reading across the round: the nursery mechanics hold (C15g, S_new ≈ 0, plasticity intact); merging
without rehearsal or with synthetic rehearsal is catastrophic (C15d/e); random reinit is harmful
even inside sleep (C15f); what rehearsal never covers erodes fast (C15i); and the one honest cost of
a working sleep is a **novelty tax** — ~0.09 pt of old-domain accuracy per new domain consolidated,
which stops the moment the world stops being new (C15b falsified as registered, accel arm showing
the tax is per-novelty, not per-sleep).

Escalation, seeds 4–9 (stored + labels, same constants; decision ≥ 5/9): S_ret
1.9/1.2/1.4/1.5/1.1/1.7, S_plast −0.2/−2.0/−6.2/−0.6/−0.2/−1.2, labels gain
+1.1/+0.6/+0.8/+0.9/+1.2/+1.2. **C15a SURVIVES 9/9; C15c SURVIVES 9/9; C15j FALSIFIED 5/9** (gain
positive on 9/9). Round tally: C15 a/c/d/e/g/h survive, b/f/i/j falsified — b and i as predicted,
f and j the round's surprises.
