# Generational consolidation agent + CAM hypervector program learner
**Dates:** Aug 20, 2026 (consolidation loop, design spec and all logged runs; CAM rounds 1–10 are not dated in the README) · **Status:** ongoing (consolidation loop has open caveats; CAM campaign closed at round 10) · **Hardware/scale:** CUDA GPU for the consolidation loop (device name not recorded; logs print `dev=cuda`), about 11 s per generation; CAM is NumPy only, no hardware stated

## Question
Does a two-weight-set loop (a frozen base serves, a learner trains on a replay pool of verified experience and is promoted only on a frozen-eval gain) accumulate interaction-derived skill across generations, and how does the density of environment grounding change that? Separately, can a gradient-free hyperdimensional learner (CAM) induce programs and learn a useful similarity metric for them?

## Approach
- **Consolidation models:** one recurrent learner per generation. GRU core with hidden size 256 (`agent/loop.py`), token embedding of (observation, action, effect-or-unknown), and a query head that predicts the effect class before the transition is absorbed. Parameter count is not stated in the sources.
- **Environment:** 9×9 gridworld (`agent/world.py`). 5 latent roles (floor, wall, goal, teleport, slide) with 12 surface appearance IDs re-permuted per world, so roles can only be identified by acting. 5×5 observation window, 4 actions, 5 effect classes. Roles are used for evaluation only.
- **Loop settings:** 24 generations, 256 episodes per generation, 200-step episodes, fixed-size sample of 2,048 pool episodes per generation, 400 training steps per generation (command given in `agent/CONSOLIDATION.md`). Batch size 16 and seed 0 are script defaults.
- **Eval:** 64 frozen worlds (seeds 900000–900063), 200 steps each, balanced accuracy over effect classes. Every generation is scored on the same seeds.
- **Promotion gate:** the learner replaces the base only if its eval beats the best eval so far. The saved checkpoint is therefore the best promoted base; `agent/control.py` evaluates these checkpoints.
- **Grounding regimes:** G1 labels every transition with the true effect. G2 labels about 10% of transitions. G3 and G4 are G1 for 6 warm-up generations, then the environment stops labelling and a judge (the model's own labels) supplies targets. G3 re-forks the judge from the base each generation; G4 freezes it at the generation-6 checkpoint.
- **Controls:** wrong-context evaluation (hidden state rolled across the batch so it summarises a different world; `control.py`), gap = eval − wrong-context. Gen-0 random base as the floor. A matched-budget i.i.d. reference (`agent/iid.py`). A promote-every-generation ablation (`--nogate`).
- **Seeds:** all logged runs use seed 0. One G1 config was rerun (`run_a_fullview.log`), see Results.
- **Pre-registered bars:** none stated.
- **CAM:** hypervectors in {−1,+1}^8192 (`cam/cam.py`, D=8192, 64 level vectors, N=8), bind/bundle/permute algebra, minimum description length as credit, anytime search over a growing library. Ten falsification rounds, each with a named attack; round 10 uses 400 sampled sequences and 30k pairs.

## Results
Consolidation values from `agent/CONSOLIDATION.md` are best-promoted checkpoints evaluated by `control.py`, not the learner's per-generation output. Gap values are as printed there and can differ by .001 from the difference of the rounded columns.

| Test | Result | Baseline / control | Verdict |
|---|---|---|---|
| G1 (full labels), best-promoted checkpoint eval, gens 1 / 4 / 8 / 12 / 16 / 20 / 24 | .281 / .413 / .540 / .622 / .622 / .630 / .630 (plateau about gen 13) | Gen 0 random base: .202 | Positive: rises and plateaus |
| G1 wrong-context gap, gen 24 | eval .630, wrong-context .395, gap +.236 | Wrong-context column is the prior-only control | Positive: the gap is interaction-derived |
| G2 (about 10% labelled), gen 24 | eval .468, wrong-context .396, gap +.073; plateau about gen 20 | G1 gap +.236 at gen 24 | Positive but weaker: gap cut 3.2x, loop still compounds |
| Wrong-context column, G1 vs G2, gen 24 | .395 vs .396 | — | Prior is the same under both regimes; grounding density changes only identification |
| G3 judge re-forked each generation, gap | +.151 at gen 6, +.179 at gen 24 | G1 gap +.236 at gen 24 | Partial: small gain after withdrawal, then flat |
| G4 judge frozen at gen-6 checkpoint, gap | +.151 at gen 6, +.171 at gen 24 | G3 +.179 at gen 24 | Null: G4 did not beat G3 (+.171 vs +.179) |
| Judge label accuracy vs truth, gens 7–24 (G3 / G4) | .579–.625 / .584–.653 (`run_s_g3.log`, `run_s_g4.log`) | Environment labelling: 1.000 | Inconclusive on usefulness; no collapse trend |
| Matched-budget i.i.d. reference on the G1 pool (6,144 episodes), step 9600 | eval .679, wrong-context .445, gap +.235; gap +.216 to +.251 across steps 1600–9600 (`run_iid.log`) | G1 learner gen 24: eval .624 (`run_s_g1.log`); checkpoint .630, gap +.236 | Negative for the schedule on raw eval (i.i.d. higher); gap about equal |
| Promote-every-generation ablation (`run_a_nogate.log`), gen 24 learner eval | .652; promoted 24 of 24 | G1 rerun `run_a_fullview.log`, gen 24 .653; promoted 16 of 24 | No visible difference in raw eval; wrong-context not measured |
| Reproducibility, same G1 config (`run_s_g1.log` vs `run_a_fullview.log`) | Identical gens 1–8; diverge at gen 9 (.556 vs .563); gen 24 .624 vs .653 | — | Run-to-run variance at a fixed seed |
| Replay count (rule 5), 32 episodes per generation | .964 balanced accuracy on replayed pool episodes vs .313 on fresh worlds (`agent/CONSOLIDATION.md`) | 256 episodes per generation, gen 12 eval .622 (`run_g1b.log`) vs .307 at 32 episodes (`run_g1.log`) | Pool memorisation with too few distinct worlds; more episodes fixes it |
| CAM round 1: extrapolate 2x past fit window | Survived 10/10 | — | Positive on the tested tasks |
| CAM round 1b: starve to 3 samples | Falsified: 90% overfit | — | Negative |
| CAM round 3: marginal × abstraction 2×2 | Survived 10/10 from 3 samples | — | Positive |
| CAM round 4: transfer to disjoint task family | Falsified: zero benefit | — | Negative |
| CAM round 5: 1 corrupted term in 8 | Exact 0/10; soft-MDL 8/10 recovers the true program | Exact MDL | Positive for soft MDL |
| CAM round 6: analogical program seeding | Falsified: zero benefit, +1700 wasted nodes | — | Negative |
| CAM round 7: is the VSA index informative? | corr(hv-sim, program-overlap) = +0.164 | — | Near noise |
| CAM round 8: hand-designed structural invariants | Falsified: +0.078, worse than naive | Naive | Negative |
| CAM round 9: learned metric (least squares, leave-one-out) | Falsified: +0.028 with 12 examples | — | Negative |
| CAM round 10: nonlinear random features, sequence similarity | corr: raw log-sequence +0.222, VSA value-phi +0.250, random Fourier (512) +0.274, echo-state +0.177; lift over chance 1.31x / 1.60x / 1.35x / 1.38x | Chance | Negative: no encoding gives a usable metric |

## What it showed
- With full labels the promotion-gated loop rose from .202 (gen 0) to .630 on a best-promoted checkpoint and plateaued near generation 13. In the G1 learner log, 14 of 24 generations were promoted.
- The wrong-context gap is what carries the claim. G1 reached +.236 at gen 24 against a wrong-context value of .395, and the wrong-context value is the same under G2 (.396). Sparse grounding (about 10% of transitions) cut the gap to +.073 and slowed the plateau. It did not stop the loop.
- Withdrawing grounding at gen 6 did not collapse the loop, but it bought little (G3 +.179, G4 +.171 at gen 24, against +.236 for G1). The predicted G4 > G3 result did not appear. Two explanations are still open: the append-only pool may be doing the anti-collapse work, and the judge may contribute nothing. Neither was tested.
- A matched-budget i.i.d. learner on the same pool reaches higher raw eval (.679 at step 9600) with about the same gap (+.235). At this scale the generational schedule appears to cost raw accuracy, not identification. This is one seed.
- CAM: credit assignment by MDL, few-shot induction, noise robustness and compositional reuse held up in the rounds that tested them. The learned similarity metric failed in four attempts (rounds 7–10). Round 10 found the signal present but faint (SNR about 1).

## Limitations
- Single seed (0) for every reported run. The G1 rerun diverged from the original from generation 9 and ended .653 against .624 at gen 24, a difference larger than the G3–G4 gap (+.179 vs +.171).
- Synthetic 9×9 gridworld with 5 roles; no comparison to other environments. 64 eval worlds, 200-step episodes.
- Consolidation results are for the best-promoted checkpoint, reported in the design spec. The per-checkpoint `control.py` outputs are not saved to a file in the project, so the wrong-context numbers can be checked only against the spec.
- The ablation run (`run_a_nogate.log`) has no wrong-context measurement, and the judge arms have no control for caveat 2 (withdrawal with a frozen pool and no new episodes).
- Caveat 1 (sliding-window pool) was not run. The residual-error sampler was not ablated against uniform sampling, so its benefit is unverified.
- The matched-budget i.i.d. run assumes the loop used `--steps 400`, as in the command in the spec; the logs do not record it.
- CAM: no output files or logs are in the project; all CAM numbers come from `cam/README.md`. Round 2 has no quantitative result. Dimension and level count are fixed at 8192 and 64.

## Key files
- `agent/CONSOLIDATION.md`: design spec, rules, G1–G4 result tables, open questions and refs.
- `agent/loop.py`: the generational loop. GRU learner, on-policy collection, judge logic for G3/G4, residual-error scoring, promotion gate, `--grounding`, `--nogate`, `--steps`, `--sample`.
- `agent/world.py`: gridworld with latent roles, surface appearance re-permutation, effect classes.
- `agent/control.py`: wrong-context control over saved checkpoints.
- `agent/sweep.py`: builds the gap, eval and judge-accuracy tables from `run/s_g*` checkpoints and `run_s_g*.log`.
- `agent/iid.py`: matched-budget i.i.d. reference trained on a run's pool.
- `agent/run_*.log`: generation logs. Columns: gen, pool size, loss, eval, late-window eval, [judge accuracy], gate, seconds.
- `agent/run/*/`: checkpoints and pools for each run (not reproduced here).
- `cam/README.md`: CAM design, falsification table (rounds 1–10) and round 10 results.
- `cam/cam.py`: CAM core. VSA algebra, MDL scoring, anytime search, D=8192.
- `cam/cam2.py`: shared module imported by the round scripts.
- `cam/r9.py`: round 9, learned metric by least squares with leave-one-out.
- `cam/r10.py`: round 10, nonlinear feature maps for sequence similarity.

## Sources
- `agent/CONSOLIDATION.md`: G1–G4 tables and gaps, judge accuracy, run command, rule 5 numbers (0.964, 0.313), rule 6, caveats, refs.
- `agent/run_g1.log`: 32 episodes per generation, gen 12 eval .307.
- `agent/run_g1b.log`: 256 episodes per generation, gen 0–12 eval, gen 12 .622.
- `agent/run_s_g1.log`: G1 24-generation learner log, gen 1–24 eval, promote count.
- `agent/run_s_g2.log`: G2 24-generation learner log.
- `agent/run_s_g3.log`, `agent/run_s_g4.log`: judge arms, judge accuracy gens 7–24.
- `agent/run_a_fullview.log`: G1 rerun (gen 24 eval .653).
- `agent/run_a_nogate.log`: promote-every-generation ablation (gen 24 eval .652).
- `agent/run_iid.log`: matched-budget i.i.d. reference, steps 800–9600.
- `agent/loop.py`, `agent/world.py`, `agent/control.py`, `agent/iid.py`: definitions and defaults (hidden size 256, batch size 16, seed 0, 64 eval worlds).
- `cam/README.md`: all CAM rounds and numbers.
- `cam/cam.py`: D=8192, LEVELS=64, N=8.
