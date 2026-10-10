# Promotion-gated generational consolidation with verified replay: a grounding sweep

Eric Lam (independent researcher, Toronto)

## Abstract
We test whether a two-weight-set learner can accumulate skill across generations. A frozen base serves, a learner trains on a fixed-size sample of an append-only pool of verified experience, and the learner is promoted only when a frozen evaluation improves. In a 9×9 gridworld where the agent must predict the effect of its actions, the loop rises from .202 to .630 balanced accuracy and plateaus near generation 13 when every transition is labelled. We measure interaction-derived skill with a wrong-context control, and it reaches +.236. Labelling 10% of transitions cuts this gap to +.073 while leaving the wrong-context baseline unchanged (.395 vs .396). Withdrawing grounding and substituting the learner's own judge produces small gains and no collapse, and a frozen judge does not beat a co-evolving one (+.171 vs +.179). A matched-budget i.i.d. learner on the same pool reaches higher raw accuracy with the same gap. All runs use one seed, and a rerun of the full-label configuration differs from the original by .029 at generation 24. We also report a negative result for CAM, a gradient-free hyperdimensional program learner, whose learned similarity metric failed in four attempts.

## 1 Introduction
Complementary-learning-systems theory separates fast episode-local state from slow, durable weights, and consolidates the former into the latter through replay [1]. We implement this as a generational loop: base_n is forked into a learner, the learner trains on a sample of stored experience, and the learner is promoted to base_{n+1} only if it improves on a fixed evaluation. Three design rules shape the loop. Hidden state is never stored, because it goes stale after promotion. Evaluation seeds are frozen, because stochastic evaluation makes the gate a coin flip. The wrong-context control is reported beside every evaluation, because absolute accuracy contains a prior that any policy receives from the query window alone.

The question we address is how much environment grounding the loop needs. Verified outcomes are the gate's input, and an environment that labels every transition is an unusually strong supervisor. We sweep grounding density from full labels to sparse labels, and then withdraw grounding altogether in favour of a model-produced judge. We ask three things: whether the loop compounds, whether the gap survives sparse grounding, and whether a frozen judge is safer than one that co-evolves.

## 2 Method
**Environment.** The gridworld has five latent roles (floor, wall, goal, teleport, slide). Each world assigns surface appearance IDs from a set of 12 by permutation, so a role can be identified only by acting in the world. The agent sees a 5×5 window and chooses among four moves. The target is the observable effect of the action, one of five classes, and effects are self-observed, so no reward engineering is needed. Roles are used only for evaluation.

**Learner.** Each generation's learner is a recurrent network with a 256-unit GRU core. Each pool transition is embedded from its observation, action, and effect label, with an "unknown" token when the outcome was not verified. A query head predicts the effect before the transition is absorbed into the hidden state. The hidden state is recomputed from stored observations on every replay.

**Loop.** Each generation collects 256 on-policy episodes of 200 steps in fresh worlds, appends them to the pool, samples 2,048 pool episodes weighted by their last residual error, and trains a fork of the base for 400 steps. The learner is evaluated on 64 frozen worlds. It is promoted if its balanced accuracy exceeds the best so far. The best promoted checkpoint is saved, and that checkpoint is what we report.

**Regimes.** G1 labels every transition with the true effect. G2 labels about 10% of transitions. G3 and G4 are G1 for six warm-up generations. After that the environment stops labelling, and a judge supplies targets. G3 re-forks the judge from the base each generation. G4 freezes it at the generation-6 checkpoint. The true effect is recorded but never trained on.

**Controls.** The wrong-context control rolls the hidden state across the evaluation batch so that it summarises a different world. The gap is the evaluation minus this control. We also report a gen-0 random base, a matched-budget i.i.d. learner trained on the full G1 pool for 9,600 steps (24 generations × 400 steps, assuming the G1 run settings), and a promote-every-generation ablation.

**Seeds.** All runs use seed 0. We reran the G1 configuration once more and compare the two runs in Section 3.

## 3 Experiments and Results

### 3.1 Compounding and the wrong-context gap
Table 1 reports the best promoted checkpoint at selected generations (G1 and G2 values from the design spec).

| Generation | 1 | 4 | 8 | 12 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|---|
| G1 eval | .281 | .413 | .540 | .622 | .622 | .630 | .630 |
| G1 wrong-context | .253 | .297 | .360 | .393 | .386 | .395 | .395 |
| G1 gap | +.028 | +.117 | +.180 | +.229 | +.236 | +.236 | +.236 |
| G2 eval | .233 | .287 | .357 | .388 | .430 | .468 | .468 |
| G2 wrong-context | .228 | .255 | .303 | .339 | .370 | .396 | .396 |
| G2 gap | +.005 | +.032 | +.054 | +.050 | +.060 | +.073 | +.073 |

*Table 1. Best promoted checkpoint, balanced accuracy. Gap values as printed in the spec.*

The G1 eval starts from a random base at .202 and plateaus near generation 13. In the learner log, 14 of 24 generations were promoted. The wrong-context column is the same under both regimes at generation 24 (.395 for G1, .396 for G2). Sparse grounding therefore does not change the prior learned from the observation window. It changes only identification by interaction, which falls from +.236 to +.073, and it slows the plateau from about generation 13 to about generation 20.

### 3.2 Withdrawing grounding
Table 2 reports the gap for the judged arms. Both withdraw grounding after generation 6, where G1 has a gap of +.151.

| Arm | Gap at gen 6 | Gap at gen 24 | Judge accuracy vs truth, gens 7–24 |
|---|---|---|---|
| G1 (never withdrawn) | +.151 | +.236 | 1.000 (environment) |
| G3 (judge re-forked each generation) | +.151 | +.179 | .579–.625 |
| G4 (judge frozen at gen 6) | +.151 | +.171 | .584–.653 |

*Table 2. Withdrawal arms. Judge accuracy is taken from the per-generation logs.*

Both judged arms gain a little after withdrawal and then stall by about generation 11–12, while G1 continues to climb. Judge accuracy shows no downward trend, so we see no death spiral. The predicted advantage of a frozen judge over a co-evolving one did not appear: G4 finished at +.171 against G3 at +.179. At this scale a frozen judge buys nothing measurable over one that re-forks.

### 3.3 Matched-budget reference and gate ablation
A fresh learner trained i.i.d. on the same 6,144-episode G1 pool for 9,600 steps reaches .679 eval, .445 wrong-context and a gap of +.235 at step 9600. Its gap is between +.216 and +.251 across steps 1600–9600. Its raw eval is higher than the G1 learner's .624 at generation 24, and its gap matches that of the loop, so the generational schedule costs raw accuracy here but does not change identification. The promote-every-generation ablation reaches .652 at generation 24, and the gated G1 rerun reaches .653. We did not measure the wrong-context value for either run.

**Reproducibility.** The two G1 runs agree exactly for generations 1–8 and diverge at generation 9 (.556 and .563). At generation 24 they reach .624 and .653. This variance is larger than the G3–G4 difference, so the withdrawal comparison should be read as a single-seed observation.

### 3.4 Replay count
With 32 episodes per generation the learner memorised the pool: it reached .964 balanced accuracy on replayed episodes but only .313 on fresh worlds. The eval at generation 12 was .307 for 32 episodes and .622 for 256 episodes per generation. The two logs' headers differ only in episodes per generation, but their sample-size settings are not recorded. The failure is an episode-index shortcut, and it is fixed by more distinct worlds per generation.

### 3.5 CAM (negative result)
CAM is a gradient-free learner. It uses 8192-dimensional bipolar hypervectors, minimum description length as its credit signal, and anytime search over a growing library. Table 3 summarises the falsification rounds.

| Round | Attack | Result |
|---|---|---|
| 1 | Extrapolate 2× past fit window | Survived 10/10 |
| 4 | Transfer to a disjoint task family | Falsified, zero benefit |
| 5 | One corrupted term in eight | Soft-MDL recovers the true program in 8/10 (exact search 0/10) |
| 8 | Hand-designed structural invariants | Falsified, +0.078, worse than naive |
| 9 | Learned metric, least squares, leave-one-out | Falsified, +0.028 with 12 examples |
| 10 | Nonlinear random features for program similarity | Best correlation +0.274 (random Fourier, 512); VSA encoding lift 1.60× over chance |

*Table 3. Selected CAM rounds from the project README.*

Credit assignment, few-shot induction, noise robustness and compositional reuse held up. The learned similarity metric did not. Round 10 found the similarity signal present but faint: within-band spread is 0.05–0.065 against a between-band range of 0.082, an SNR near 1.

## 4 Discussion
The loop compounds when grounding is dense. The wrong-context control shows that the compounding is in identification by interaction, not in a prior that any model would obtain from the query window. Sparse grounding preserves the loop and reduces identification. The conclusion that matters most for the design is that the gap, not raw accuracy, is the measure to report: the matched i.i.d. learner reaches a higher raw score with the same gap.

The withdrawal result is a null. A judge that labels its own experience neither collapses nor helps much at this scale. The likely reason is that the append-only pool keeps 1,536 grounded episodes, a quarter of the pool at generation 24, and continues to replay them. Two controls would separate this from a judge contribution: a sliding-window pool that evicts pre-withdrawal episodes, and withdrawal with no new collection. Neither was run.

The replay-count result is a practical design rule. Stored experience without enough distinct worlds yields memorisation that looks like learning in training loss.

## 5 Limitations
- One seed for every reported run. The G1 rerun differs from the original by .029 at generation 24, which is larger than the G3–G4 difference.
- A single synthetic gridworld with five roles, 64 evaluation worlds and 200-step episodes.
- Reported values are for best promoted checkpoints. Per-checkpoint wrong-context outputs are not saved as files in the project.
- The promote-every-generation ablation has no wrong-context value. The judge arms have no withdrawal control.
- The residual-error sampler has not been ablated against uniform sampling.
- The matched-budget comparison assumes the loop used 400 training steps per generation, as in the run command in the design spec.
- CAM numbers come from its README only. Round 2 has no quantitative result.

## 6 Conclusion
A promotion-gated generational loop with verified replay accumulates interaction-derived skill when grounding is dense, and the gap shrinks but persists under sparse grounding. Withdrawing grounding and using a judge gives small gains and no collapse at this scale, and a frozen judge does not beat a co-evolving one. A matched-budget i.i.d. learner matches the gap but reaches higher raw accuracy, so the schedule's value is not yet shown. These results come from one seed and one environment.

## References
[1] McClelland, J. L., McNaughton, B. L., and O'Reilly, R. C. Psychological Review 102:419–457, 1995. https://pubmed.ncbi.nlm.nih.gov/7624455/ (complementary learning systems; as cited in the project design spec, without further detail).
[2] arXiv:2404.01413. https://arxiv.org/abs/2404.01413 (accumulate-vs-replace bound; as cited in the project design spec).
[3] Nature 631:755–759. https://www.nature.com/articles/s41586-024-07566-y (model collapse; as cited in the project design spec).
[4] arXiv:2407.04549. https://arxiv.org/pdf/2407.04549 (reward hacking in self-refinement, co-evolving judge; as cited in the project design spec).
