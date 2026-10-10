# Small adaptive systems: can compact learners improve through experience under fixed limits?
**Dates:** Not stated in the docs (source docs last edited Sep 8–9) · **Status:** concluded (development pilots only; no confirmatory study) · **Hardware/scale:** One RTX 4080 computer named in the charter; Windows desktop. Pilots ran on CPU (a GPU smoke test was slower). Controllers: 1,060 registered learned parameters (neural pilots); 40 online coefficients (Idea 02 model bank); at most 74 scalars (Idea 03 predictor).

## Question
Can a compact learner that adapts online improve its control of an unfamiliar simulated system from its own experience, while model size, state and compute stay fixed and earlier competence is kept? All tests are on simulated systems; no real device was controlled.

## Approach
- **Task:** simulated control with two sensor values, two bounded actions and a target. Neural pilots switch dynamics A → B → A in 32-step phases inside 96-step episodes. Development families: familiar linear dynamics, longer action delay, cross-channel coupling, nonlinear damping. Idea 02 used the same four families with 96 interactions per environment. Idea 03 used a separate NumPy world (192 interactions per episode) with linear dynamics plus three absent effects: actuator dead zone, actuator saturation, state-dependent action gain.
- **Neural arms (Pilots 001–002):** 32-unit recurrent controller with 1,060 registered learned parameters. Three modes share initialization and candidate sources: fixed (no fast weight or routing updates), plastic (weight updates only), rewire (weight and bounded connection-selection updates; 8 candidate sources per unit, 4 active). Offline training: 150 optimizer updates × 32 episodes × 96 interactions = 460,800 transitions per model, by backpropagation through the training simulator. The deployed controller sees only observations and targets.
- **Online predictive models (Idea 02):** four linear models, one per action delay 0–3, with model weights from discounted prediction error. A single all-lag model (22 fitted coefficients) is the comparison; the bank fits 40 coefficients. Ridge fits use the latest 16 transitions. Planning scores 25 constant actions over an 8-step horizon. No offline training.
- **Learned features (Idea 03):** a ridge linear predictor over 10 scalar inputs, with up to four learned tanh features proposed every 16 observations. The gated variant keeps a candidate only if it lowers accumulated error by at least 5% over the next 8 observations and gives a mean absolute MSE improvement above 1e-4.
- **Controls and baselines:** zero action; proportional feedback; recursive least squares (RLS) identification with a horizon-six controller; fixed neural controller (three saved seeds); passive bank (Idea 02 with information bonus disabled); single all-lag model; linear predictor; four fixed random features; ungated learned features.
- **Seeds:** neural pilots 11, 23, 37. Idea 02 seed bases 781001 (original development), 881001 (fresh draws), 981001 (random hidden-change timing and targets). Idea 03 feature seeds 11, 23, 37 with seed bases 1081001 (development) and 1181001 (fresh draws).
- **Pre-registration and bars:** Idea 02 and Idea 03 wrote protocols before their results and froze their choices. Idea 03 fixed its gate thresholds (5% and 1e-4) in advance. Pilots 001–002 and Experiment 01 were not preregistered, and their docs state that no confirmatory claim is licensed.

## Results
| Pilot | Result | Baseline / control | Verdict |
|---|---|---|---|
| Pilot 001, rewire: tracking MSE (familiar / longer delay / coupling / nonlinear) | 0.1005 / 0.1416 / 0.1416 / 0.1281 | Fixed 0.1000 / 0.1414 / 0.1418 / 0.1276 | No consistent advantage over fixed |
| Pilot 002, normalized routing rewire: same four families | 0.0990 / 0.1412 / 0.1426 / 0.1285 | Fixed 0.1000 / 0.1414 / 0.1418 / 0.1276 | Familiar mean about 0.95% better, but one of three training seeds is worse; coupling and nonlinear slightly worse |
| Neural pilots vs RLS (same families) | Fixed 0.1000 / 0.1414 / 0.1418 / 0.1276 | RLS 0.1978 / 0.2820 / 0.2663 / 0.2176; zero action 0.1653 / 0.1755 / 0.1800 / 0.1718 | Lower MSE than RLS, but on familiar environments within 0.1 of target on 9.8% of steps (fixed), 8.1% (Rewire 002) and 45.1% (RLS): a tradeoff, not universal superiority |
| Pilot 002 ablation: online weight and routing updates disabled | Normal 0.0990 / 0.1412 / 0.1426 / 0.1285; disabled 0.1125 / 0.1653 / 0.1662 / 0.1353 | Separately trained fixed network performs similarly to the normal model | Trained model depends on its updates; not a demonstrated comparative benefit |
| Pilot 002, state reset on return to A: return-phase MSE | Familiar 0.1087 → 0.1081; delay 0.1387 → 0.1402; coupling 0.1446 → 0.1445; nonlinear 0.1514 → 0.1459 | Normal (no reset) | Mixed; not evidence of reliable retention (reset also removes hidden state) |
| Idea 02, bank + information: MSE (original / fresh / random timing) | 0.1050 / 0.1124 / 0.1017 | Fixed neural 0.1277 / 0.1184 / 0.1225; original RLS 0.2409 / 0.2006 / 0.1817 | Lower than fixed neural in all three suites. Paired differences: −0.0228 [−0.0359, −0.0099], −0.0059 [−0.0202, +0.0079], −0.0208 [−0.0294, −0.0115]. Fresh-draw interval includes zero |
| Idea 02, single all-lag model (same suites) | 0.1139 / 0.1184 / 0.1005 | Bank + information 0.1050 / 0.1124 / 0.1017 | Bank is not universally superior; single model is lowest in random timing |
| Idea 02, longer-delay family, fresh draws | Bank + information 0.1281 | Fixed neural 0.1137; passive bank 0.1196; single model 0.1560 | Clear failure; fixed neural wins this family in all three suites |
| Idea 02, information bonus vs passive bank: MSE difference (original / fresh / random timing) | −0.0050 / +0.0033 / −0.0013 | Passive bank 0.1099 / 0.1092 / 0.1030 | Inconsistent; every paired interval includes zero. Bonus changed actions on 40.3% of steps but did not clearly help |
| Idea 03, gated learned features, fresh control MSE | 0.05616 | Linear 0.06024; fixed random features 0.05674; ungated learned 0.05508 | 6.8% below linear (paired −0.00408, [−0.00774, −0.00085]). Versus fixed random (−0.00058, [−0.00783, +0.00881]) and ungated (+0.00108, [−0.00419, +0.00712]): uncertain. No established advantage over controls |
| Idea 03, development control MSE | Gated 0.06064 | Linear 0.06211; fixed random 0.06092; ungated 0.05425 | Gated vs linear −0.00147 [−0.00557, +0.00212]: uncertain |
| Idea 03, fixed-action prediction probe (same observations and actions) | Gated 0.010032 | Linear 0.009901; fixed random 0.010496; ungated 0.010599 | Negative: gated worse than linear by +0.000131 [+0.000046, +0.000221]; control gain does not show better prediction |

## What it showed
- Online adaptation changed behavior, but rewiring at this scale gave no consistent tracking gain over a fixed recurrent network of the same size. Pilot 002 raised structural activity from about 0.02 to 4.25–4.50 replaced connections per decision without a consistent benefit. The pilots stopped after two versions.
- Explicit online prediction with a bank of four delay models (Idea 02) lowered error relative to the fixed neural controller in all three suites, with no offline training. It failed on longer action delays, and the information bonus did not clearly beat the passive bank.
- Learned nonlinear features (Idea 03) lowered fresh control error by 6.8% relative to a linear predictor. They did not clearly beat fixed random features or ungated learning, and they predicted worse on a shared probe. The feature gain is therefore not evidence of better explanations.
- Lower error and reaching the target within tolerance did not move together. RLS had the highest error but was within tolerance on far more familiar-environment steps (45.1%) than the neural controllers (8.1–9.8%). Neither measure is a settled success criterion, because target reachability was not verified.
- Across all mechanisms, the docs record no reproducible advantage over strong fixed or simpler controls. The project decisions say so: the rewiring pilot is stopped, the information bonus is not adopted by default, and the gate is not promoted.

## Limitations
- Simulated systems only. Phone performance, energy use and real-device usefulness were not measured. Latencies are short desktop measurements.
- Dynamics vocabulary, feature function class, action lags and the stable-dynamics prior were supplied by the designer. Idea 02 and Idea 03 evaluated on generators designed with knowledge of the original simulator.
- Targets are not guaranteed reachable under actuator limits, so tolerance frequencies are descriptive, not task-success rates. Idea 03 verified steady-state witnesses, not immediate attainability.
- Pilot 002 used development evaluation data that had already been inspected. Pilots 001–002 and Experiment 01 were not preregistered.
- Bootstrap intervals resample environments within families only. They omit training-seed and family-selection uncertainty and are not corrected for multiple comparisons.
- Pilot 001 and 002 had only three training seeds per neural mode, and the seed-level result for the 0.95% familiar improvement shows one worse seed.
- The state-reset retention check confounds hidden-state removal with adaptive-state removal, and fixed phase timing and repeated targets confound any memory reading.
- The PyTorch evaluation process peaked at 487.8 MiB; that is a development measurement, not the exported runtime.

## Key files
- `adaptive/model.py`, `adaptive/world.py`, `adaptive/train.py`, `adaptive/evaluate.py`, `adaptive/numpy_runtime.py`: active neural pilot source (model, simulator, offline training, evaluation, standalone NumPy runtime), per README.md.
- `adaptive/export.py`, `adaptive/demo.py`: export a trained controller to a NumPy bundle and run it on a simulated system.
- `adaptive/hypotheses.py`, `adaptive/hypotheses_demo.py`, `adaptive/evaluate_hypotheses.py`: Idea 02 online model bank, demo and full comparison.
- `adaptive/features.py`, `adaptive/features_demo.py`, `adaptive/evaluate_features.py`: Idea 03 gated and ungated learned features, demo and full comparison.
- `results/adaptive/evaluation-001/evaluation.json`, `results/adaptive/evaluation-002/evaluation.json`: raw neural pilot results.
- `results/adaptive/pilot-002/training_summary.json`: Pilot 002 training record.
- `results/hypotheses/pilot-001/evaluation.json`, `results/hypotheses/pilot-001/aggregate.json`: Idea 02 per-environment and paired-interval results.
- `results/features/pilot-001/evaluation.json`, `results/features/pilot-001/aggregate.json`, `results/features/pilot-001/protocol.json`: Idea 03 results, paired intervals and frozen protocol with source hashes.
- `requirements-adaptive.txt`: pilot Python environment.
- `tests/`: unit tests (docs report 18, 22 and 27 passing for the three implementations respectively).

## Sources
- `README.md`
- `CHARTER.md`
- `EXPERIMENT_01.md`
- `UPDATE_RULE.md`
- `PILOT_RESULTS.md`
- `IDEA_02.md`
- `HYPOTHESIS_RESULTS.md`
- `IDEA_03.md`
- `FEATURE_RESULTS.md`
