# Decision model (LCM): one trained scorer for answers and memory/tool actions
**Dates:** Sept 19–20, 2026 (decision-model documents; repository work began Sept 17, 2026) · **Status:** Internal R&D prototype. A CPU API runs on a published checkpoint. A research preview also runs, but it failed a retention gate. Broader candidates were not promoted. The general-reasoning goal is unmet. · **Hardware/scale:** Training on a local NVIDIA RTX 4080 SUPER (GPU released by the user for these runs). Serving on CPU with two threads. Synthetic and small public corpora, with 42,562 to 60,176 atomic training rows depending on the run.

## Question
Can one shared scorer, built on a pretrained text encoder, choose typed answers (choice, ordered score, truth probability) and memory/tool actions from caller-supplied candidate descriptions? If so, does it hold up on unseen wording, structures and environments without losing earlier skills?

## Approach
- **Models.** A shared pretrained encoder reads state tokens. A multihead attention layer reads them for each candidate. A shared MLP scalar head scores candidates, and results are normalized within each question. No text is generated; question and option IDs are opaque routing fields (`lcm/decision_model.py`, `lcm/decision_io.py`).
  - Backbone 1: `all-MiniLM-L6-v2`, 22.89M total parameters (22,894,593 measured in `docs/decision-preparation-2026-09-19.md`). The R&D plan estimated about 22.7M.
  - Backbone 2: a pretrained NLI DeBERTa encoder with a new scalar scorer, 141.53M parameters. The original NLI classification head was discarded.
- **Training data.** Synthetic atomic families (logic conditions, ordering, lookup, events, routing, severity, helpdesk next action). Public WANLI NLI pairs (30,000 balanced pairs in the generalization run; 12,000 three-way records in the broad run). BoolQ and QuaRTz reading and causal items (8,786 training records in the grounded corpus). Memory workflows over arithmetic dependency graphs with supplied tools for arithmetic, freshness checks and invalidation. Reference planners generated demonstrations offline only; inference never called them.
- **Training regimes.**
  - Supervised (accepted-action labels) for atomic and workflow decisions.
  - Retention ablation: identical runs, one with a cached-reference KL term (weight 0.5, temperature 4) and one control.
  - Supervised (SFT) versus PPO from the same warm start: 192 supervised warm-start updates, then 512 episodes and 256 optimizer passes per arm. Rewards were +1 for a verified success, −1 for a wrong conclusion or exhausted budget, and −0.01 per action.
- **Calibration.** Temperature was fit on separate calibration splits and never on test data.
- **Gates.** Development selection came before reserved tests. The predeclared retention gate allowed at most a 5-point drop on any original-family or helpdesk exact score relative to the published model.
- **Game experiments.** None in this log. The repository's arcade and Minesweeper experiments are separate work.

## Results
| Test | Result | Baseline / control | Verdict |
|---|---|---|---|
| Six trained atomic families, new wording and names (reserved) | 371/384 (96.61%) | Matched pretrained similarity 160/384 (41.67%) | Positive, in-distribution |
| Untrained conjunction/prerequisite families (reserved) | 37/128 (28.91%) | None stated | Negative: low transfer |
| All eight atomic families (reserved) | 408/512 (79.69%) | None stated | Mixed: strong on trained families only |
| API-default workflows (reserved, 24) | 12/24 (50%) | Random actions 0/24 | Above random; not reliable |
| Workflows without name compilation (reserved, 24) | 17/24 (70.83%) | Same setup with API defaults | Positive; name compilation reduced aggregate success |
| API-default workflows under perturbations | Clean 3/6, timeout 3/6, changed input 5/6, combined 1/6 | None stated | Negative under combined disturbance |
| Guided helpdesk, exact sequence (exposed, 44 correlated variants) | 28/44 (63.64%) | None stated | Not independent evidence |
| Guided helpdesk, correct team/handoff with sufficient evidence (exposed) | 33/44 (75%) | None stated | Not independent evidence |
| Temperature calibration, NLL on calibration split | 1.6476 → 0.4116 | Uncalibrated 1.6476 | Positive on that split; final atomic NLL 0.5140, Brier 0.2938, ECE 0.1023 |
| Shared scorer, dense logic practice: MiniLM vs DeBERTa, development | 144/256 (56.3%) vs 209/256 (81.6%) | MiniLM 1.621 NLL vs DeBERTa 2.099 NLL | Mixed: higher accuracy, worse probability quality |
| Broader MiniLM run, reserved evaluation: three-way NLI | 194/600 (32.33%) → 269/600 (44.83%) | Published baseline 194/600 | Positive, but most cases still wrong |
| Same run: six original families (retention) | 183/192 (95.31%) → 161/192 (83.85%) | Published baseline 183/192 | Negative: material regression |
| Same run: severity classification | 32/32 → 15/32 | Published baseline 32/32 | Negative |
| Same run: Boolean conditions, balanced accuracy | 48.53% | Baseline 50.49%; always-false raw score 85% | Null; raw 82.50% was misleading |
| Matched retention, control arm (2,400 updates) | Routing −31.25 points; lookup −9.375 points | Published model | Negative; fails gate |
| Matched retention, KL arm (2,400 updates) | Original six families 176/192; severity −21.875 points | Published 183/192; parent 161/192 | Negative; fails gate (not promoted) |
| Broad DeBERTa run, reserved: new memory workflows | 48/48 | Published 23/48 | Positive |
| Broad DeBERTa run, reserved: ordering | 100% | Published 26.7% | Positive |
| Broad DeBERTa run, reserved: grounded yes/no / qualitative causal | 74.0% / 83.9% | Published 51.7% / 47.2% | Positive |
| Broad DeBERTa run: original ticket routing (retention control) | 13/32 | Published 32/32 | Negative: −59.4 points; not promoted |
| SFT vs PPO, four adapted environments (48 episodes each) | SFT 149/192 (77.6%); PPO 133/192 (69.3%) | Common warm start 102/192 | PPO below SFT under this budget; single seed |
| SFT vs PPO, diagnostic transfer (48 episodes) | 0/48 for all four models | Broad parent 0/48 | Negative |
| Research preview, supplemental cohort | 152/240 total, of which 77 were correct abstentions | Previous repair 153/240 | Mixed |
| Research preview, document environment | 45/48 (previously 48/48) | Retention allowance 5 points | Failed gate (6.25-point drop) |
| Authored free-form decision probes (3) | Preview 0/3 | Baseline 2/3 | Negative |

## What it showed
- A shared scorer with caller-supplied candidates learned the trained atomic families well on new wording and names (371/384). It did much better than a pretrained similarity baseline on the same families (41.67%). It did not transfer to untrained conjunction and prerequisite families (28.91%).
- Multi-step memory and tool workflows improved substantially on reserved cases after the broader runs. Those gains came with regressions. Ticket routing dropped 59.4 points in the broad run, and a matched KL retention term did not pass the gate. No candidate was promoted, so the published API is unchanged.
- Accuracy and probability quality disagreed. The larger DeBERTa scorer was more accurate on the dense logic diagnostic (81.6% versus 56.3%), but its negative log likelihood was worse (2.099 versus 1.621). Temperature calibration improved the calibration split, but it does not fix wrong answers on new domains.
- PPO did not beat supervised corrective examples under this single-seed, fixed-budget comparison (133/192 versus 149/192). This does not show that RL cannot help. All four models failed the diagnostic environment by concluding without gathering evidence (0/48 each).
- The overall result is engineering progress on structured tasks, not evidence of general reasoning. Failures are preserved in the saved results, including the research preview's failed gate and its three failed free-form probes.

## Limitations
- The corpora are synthetic or small public sets with repeated templates. The routing training set contained only four distinct problem wordings. Row counts overstate the number of independent examples.
- Many "reserved" controls became exposed after evaluation and were then reused for development. The documents mark them as non-fresh evidence. The 44 helpdesk variants share 14 ticket sentences and are correlated, not independent incidents.
- Most results are single-seed. The DeBERTa runs used different pretraining, initialization, pooling and parameter count than MiniLM, so the architecture effect is not isolated.
- Possible pretraining overlap cannot be excluded for MiniLM (AllNLI) or for the NLI backbone (SNLI, MultiNLI, WANLI prompt provenance).
- The WANLI data has balanced class priors. Binary and three-way records are correlated, so statistical claims are limited.
- Calibration fits one temperature on separate splits. It does not guarantee correctness on new domains, and per-action probabilities are not workflow-success probabilities.
- Tool results, freshness checks and exact arithmetic are supplied by the runtime. The model does not learn these.
- A static review of an external PPO reference implementation (not run) found target-bearing features and reused validation data. The project did not copy that setup.
- The reference NLI classifier used for diagnostics has 141,897,219 parameters, a different head and different pretraining from the 141.53M shared scorer.
- Serving is CPU-only on a two-thread configuration. CPU and GPU timings are not directly comparable.

## Key files
- `lcm/decision_io.py`: versioned Choice, ordered Score and truth-probability input and output contract.
- `lcm/decision_model.py`: `DecisionCore`. A shared encoder, state-to-candidate multihead attention, and a shared scalar scorer.
- `lcm/decision_data.py`: frozen train, development, calibration and test corpora with separate names and wording.
- `lcm/decision_memory.py`: dependency memory, typed tool workflows, changing sources, timeouts, and the autonomous rollout path that never calls the reference planner.
- `lcm/decision_train.py`: atomic and joint training stages, development selection, checkpoint and RNG resume state, and temperature calibration.
- `lcm/decision_eval.py`: accuracy, macro family accuracy, NLL, Brier score, ECE, confidence/coverage and autonomous episode evaluation.
- `lcm/decision_api.py`: local HTTP API with explicit validation and errors. It refuses to serve untrained heads.
- `lcm/decision_helpdesk.py`: IT ticket triage and diagnostic next-action demo.
- `lcm/decision_grounded.py`: BoolQ and QuaRTz adapter with grouped, leakage-controlled splits.
- `lcm/decision_curriculum.py`, `lcm/decision_generalization.py`, `lcm/decision_supervision.py`, `lcm/decision_application_eval.py`: curriculum construction, generalization corpora, supervision labels, and application-level evaluation.
- `lcm/decision_ui.html`: browser demo.
- `scripts/decision_rnd/run_retention_ablation.py`, `evaluate_retention_ablation.py`: matched KL versus control retention experiment.
- `scripts/decision_rnd/run_feedback_experiment.py`, `feedback_environments.py`, `feedback_learning.py`, `evaluate_feedback_experiment.py`: SFT versus PPO pilot and its four environments.
- `scripts/decision_rnd/evaluate_broad_shared.py`, `continue_broad_shared.py`: broad shared-model evaluation and continuation.
- `docs/decision-api-guide.md`: API endpoints and runbook.

## Sources
- `docs/decision-training-2026-09-19.md`: published baseline results, experiment history and architecture revision.
- `docs/decision-learning-results-2026-09-19.md`: MiniLM and DeBERTa dense-practice diagnostic, with parameter counts.
- `docs/decision-generalization-results-2026-09-19.md`: broader run, failure audit and calibration.
- `docs/decision-retention-results-2026-09-19.md` and `docs/decision-retention-protocol-2026-09-19.md`: matched control versus KL retention experiment and gate.
- `docs/decision-broad-results-2026-09-20.md`: DeBERTa broad run with routing regression; not promoted.
- `docs/decision-broad-baseline-2026-09-19.md`: published-model development baseline.
- `docs/decision-feedback-results-2026-09-20.md` and `docs/decision-feedback-pilot-protocol-2026-09-20.md`: SFT versus PPO pilot and research preview.
- `docs/decision-preparation-2026-09-19.md` and `docs/decision-rnd-plan-2026-09-19.md`: preparation audit, parameter count, plan and pass criteria.
- `docs/decision-grounded-data-2026-09-19.md`: BoolQ and QuaRTz corpus.
- The 2026-09-20 reference-implementation review note in `docs/` (filename omitted here because it names an outside system): static review of a reference PPO implementation.
- `PROJECT_STATE.md` and `HANDOFF.md`: status as of Sept 20, 2026.
- External: WANLI (generated and human-labeled NLI pairs), BoolQ, QuaRTz, `all-MiniLM-L6-v2` model card, PPO (arXiv:1707.06347), GAE (arXiv:1506.02438).
