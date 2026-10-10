# Fruit-fly connectome networks (flynet) and structured sparse recurrence (SRCN)
**Dates:** 2026-09-12 (SRCN manifests and research notes; flynet results and working notes) · **Status:** paused (exploratory; the docs list next experiments that were not run) · **Hardware/scale:** SRCN quick suite on CPU, 1 thread, Windows 10, PyTorch 2.11.0+cu128, 64 recurrent units (manifest); flynet hardware not stated in its docs; flynet connectome subgraph of 2,000 neurons and 178,570 edges (logs)

Path convention in this log: `flynet/` is the flynet repository root and `srcn/` is the SRCN repository root (the SRCN code lives in `srcn/srcn/`, the flynet package in `flynet/flynet/`).

## Question
Does the wiring of a real fly nervous system make a recurrent network learn faster than degree-matched random wiring? The SRCN prototype asks a related question with synthetic module-and-hub topology: does structured sparse recurrence help small models learn memory, ordering and online reward tasks, compared with random and degree-preserving controls?

## Approach
- **Connectome source (flynet):** MaleCNS v1.0 flat-connectome files (neuron annotations, consensus neurotransmitter sign, synapse-count weights). The main experiment uses the olfactory-receptor-to-descending-neuron subgraph of 2,000 neurons and 178,570 edges.
- **Flynet models:** `real` (MaleCNS wiring, signed by neurotransmitter); `shuffled` (edge targets permuted, degree-matched); `random` (same neurons and edge count, uniform wiring); `dense` (ordinary RNN, about the same edge count); `gru` (about the same edge count). Later branches add typed shared dynamics (`typed`, `typed_shuffled`, `typed_randtypes`), a frozen connectome with a trained interface (`frozen`, `frozen_shuffled`), and reward-gated plasticity at Kenyon-cell to MBON synapses with optional dopamine-neuron gating.
- **Flynet tasks:** odor navigation in a 12×12 grid arena with 8 obstacles and horizon 64 (learning-efficiency experiment); odor conditioning, where one of two odors is food, with 24 trials per episode (branches B and C); a supervised adding problem for the structured-mask experiment (D).
- **SRCN models:** `srcn` (synthetic modules and hubs), `random_sparse` (same width and edge count), `rewired_sparse` (directed edge swaps that preserve every node's in- and out-degree), `dense_rnn` (all off-diagonal recurrent connections), `matched_dense` (width chosen to match total parameters), `gru` (width chosen to match total parameters). Hidden size 64, density 0.125.
- **SRCN supervised tasks:** delayed recall (remember one of 8 symbols through 4 distractors; 12 distractors out of distribution); rule switching (remember the latest cue, select one of two query bits, 4 distractors; 12 out of distribution); transitive ordering (shuffled adjacent facts from 6-node chains; 8-node chains out of distribution). A 25% sensor-loss condition removes input channels for a whole episode.
- **SRCN reward task:** four cues, two actions, scalar immediate reward, unannounced reversal halfway through; a frozen recurrent core with action-value readouts updated by `delta = reward - Q(h, a)` on the chosen action; epsilon 0.1; 1,500 steps per run with 32 pooled streams; a learning-disabled SRCN control.
- **Plasticity rules:** SRCN reward learning is a local delta-rule on readout weights (immediate reward, gamma 0, no eligibility trace). An earlier actor-critic pilot is kept in `reward-quick` and is not the current learner. Flynet uses a three-factor rule (presynaptic activity, chosen action, prediction error) at Kenyon-cell to MBON synapses.
- **Training (SRCN supervised, quick preset):** 200 optimizer steps, batch 32, learning rate 0.003, 6,400 training episodes per run, 3 seeds (0 to 2). Seeds are paired across models within a task.
- **Seeds:** SRCN: 3 seeds throughout. Flynet: `le_*` files hold 8 seeds (s0 to s7) for most agents; odor-conditioning runs mostly use single seeds or 4 seeds; the structured-mask experiment is a single run per mask in the CSV. The flynet working notes mention a 41-seed sweep; the repository does not contain a summary file that reproduces that count.

## Results
All SRCN values are mean ± sample SD across 3 seeds, in percent, copied from `srcn/results/baselines-quick/report.md` or `srcn/results/reward-online-q/report.md` unless marked otherwise. Flynet odor-conditioning values are the "late-trial accuracy" printed in each run's log (chance 0.50).

| Test | Result | Baseline / control | Verdict |
|---|---|---|---|
| SRCN delayed recall, in-distribution accuracy | srcn 92.8 ± 3.1 | random_sparse 92.4 ± 2.9; rewired_sparse 91.3 ± 4.7; dense_rnn 100.0 ± 0.0 (5,320 params); gru 97.4 ± 4.5 | Null for topology; dense RNN highest |
| SRCN delayed recall, 12 distractors (OOD) | srcn 46.2 ± 11.3; paired SRCN minus rewired +5.99 (SD 15.05) | random_sparse 44.4 ± 13.8 (paired +1.82, SD 6.72); rewired 40.2 ± 4.1; dense_rnn 95.4 ± 5.6; gru 76.8 ± 32.7; matched_dense 81.6 ± 11.5 | Null; SD larger than the paired mean |
| SRCN delayed recall, 25% sensor loss | srcn 70.6 ± 4.1 | random_sparse 72.1 ± 2.8 (paired SRCN minus random -1.56); rewired 70.4 ± 2.6 | Null |
| SRCN rule switching, in-distribution | srcn 81.9 ± 3.5 | random_sparse 80.1 ± 1.4; rewired 81.8 ± 3.3; dense_rnn 97.7 ± 1.4; gru 83.3 ± 7.8; matched_dense 79.3 ± 1.8. Memory-free reference 75% | Null; sparse models sit near the memory-free level, so rule memory is not established |
| SRCN rule switching, 12 distractors (OOD) | srcn 75.5 ± 3.2 | random_sparse 77.3 ± 1.0 (paired SRCN minus random -1.82, SD 3.63); rewired 76.8 ± 1.3; dense_rnn 63.7 ± 5.3; gru 78.1 ± 3.4 | Null |
| SRCN transitive ordering, in-distribution | srcn 47.9 ± 1.2 | random_sparse 48.8 ± 4.3; rewired 48.0 ± 2.7; dense_rnn 47.3 ± 3.2; gru 49.5 ± 0.6; chance 50 | Negative: all models at chance |
| SRCN transitive ordering, 8-node chains (OOD) | srcn 48.6 ± 4.7 | random_sparse 48.2 ± 2.5; rewired 49.1 ± 3.2; dense_rnn 50.7 ± 4.4; gru 46.6 ± 3.8 | Negative: all models at chance |
| SRCN online reward, recovery after reversal | srcn 94.8 ± 0.2 | random_sparse 94.9 ± 0.2; rewired_sparse 94.9 ± 0.2; matched_dense 94.1 ± 1.0; learning-disabled srcn 50.3 ± 0.9 | Positive for learning; no topology advantage |
| SRCN online reward, immediately after reversal | srcn 7.3 ± 3.2 | random_sparse 5.3 ± 0.3; rewired_sparse 5.3 ± 0.3; matched_dense 10.7 ± 9.2; learning-disabled 50.2 ± 0.4 | Null across sparse models |
| SRCN actor-critic pilot (`reward-quick`), recovery after reversal | srcn 49.5 ± 0.4 | random_sparse 49.5 ± 0.4; rewired_sparse 49.8 ± 0.2; matched_dense 49.8 ± 0.6 | Negative: pilot did not recover; not the current learner |
| Flynet odor conditioning, real wiring, plastic (`oc_real.log`, default run) | 0.74 | Chance 0.50; `oc_plastic_shuffled.log` 0.51; `oc_dense.log` 0.51 | Positive in one run; not replicated across seeds (see below) |
| Flynet odor conditioning, real wiring, seeds 1 to 3 (`oc_real_s1_u600_t24`, `oc_real_s2_u600_t24`, `oc_real_s3_u600_t24`; sampled) | 0.65, 0.71, 0.61 | Same budget: no matched shuffled seeds in the repository | Positive, modest |
| Flynet odor conditioning, GRU (`oc_gru.log`, `oc_gru_s0_u2000_t24` to `oc_gru_s3_u600_t24`) | oc_gru 0.72; seeds 0 to 3: 0.89, 0.99, 0.78, 0.76 (sampled) | Default run: real 0.74 (`oc_real.log`). Seed runs use different update budgets, so not directly comparable with the real-wiring seeds | Default runs close (0.72 vs 0.74); GRU higher in the longer-budget seeds |
| Flynet frozen anatomy, with and without rule (`oc_anat`, `oc_anat_ruleoff`) | oc_anat seeds 0 to 2: 0.74, 0.50, 0.46; oc_anat_ruleoff seeds 0 to 2: 0.49, 0.50, 0.74 (greedy 0.77 for seed 2) | Rule-off control | Null: rule on and rule off overlap across seeds |
| Flynet frozen sparse with and without trained plasticity (`oc_plastic_frozen_sparse`, `oc_plastic_frozen`) | oc_plastic_frozen_sparse seed 0: 0.74; seeds 1 to 3: 0.50, 0.72, 0.69 | oc_plastic_frozen 0.49 | Mixed |
| Flynet dopamine-gated variants (`oc_dan_frozen_sparse`, `oc_dan_frozen_sparse_shuffled`, `oc_dan_gain_frozen_sparse`) | 0.61; 0.49; 0.51 | Shuffled and gain-only variants | Weak; single runs |
| Flynet frozen-connectome control with zero learning rate (`oc_control.log`) | late_acc 0.51 | Chance 0.50 | Null control |
| Flynet learning efficiency, real vs shuffled vs random vs dense (`le_*` CSVs; from working notes) | Notes: real wiring "ties dense (7/8, median reward 13.9)"; shuffled 0/8 learned; frozen untrained beats shuffled 4/8 vs 0/8 | Shuffled wiring; dense RNN | Unverified: the learned criterion and median are not defined in code or in any summary file in the repository |
| Flynet structured-mask adding problem, final test MSE at step 3000 (`structured_sparsity.csv`, one run per mask) | fly 0.00043663440737873316; fly_signed 0.0591934472322464; degree_matched 0.002025195863097906; block_model 0.0035746425855904818; random 0.0012879511341452599; dense 0.21438506245613098 | Degree-matched, block-model, random, dense masks | Single run; conflicts with the working-note conclusion of no supervised advantage (see below) |
| Flynet plastic-synapse claim (working notes) | Withdrawn in the notes ("retracted plastic-synapse claim") | Not reproduced in any file | Negative: no written retraction exists in the repository |
| Flynet compartment-gated dopamine (working notes, seed 0) | Notes: 0.74 real, 0.55 shuffled, 0.49 rule-off | Not located in `flynet/results/` | Unverified |

## What it showed
- **Supervised SRCN:** no consistent advantage for structured topology over random or degree-preserving rewired controls. Paired differences are small and the seed SDs are as large as the means (for example the OOD recall paired difference against rewired is +5.99 with SD 15.05).
- **Parameter count dominates:** the dense RNN was the strongest model on in-distribution and OOD recall, but it had three to four times the parameters of the sparse models (5,320 vs 1,792 on recall; 4,610 vs 1,082 on rule switching). The comparison at matched budgets is weak.
- **Memory and ordering tasks were mostly not solved:** rule switching sits near the 75% memory-free reference for the sparse models, and transitive ordering is at chance for every model.
- **Online reward:** all sparse and matched-dense learners recovered to about 94 to 95% after an unannounced reversal, while the learning-disabled control stayed at chance. This shows that online readout learning works in this easy task, but not that the topology helps. The actor-critic pilot did not recover.
- **Fly-derived wiring:** the logged results are mixed and do not support a clear connectome advantage. Real-wiring odor-conditioning runs reached 0.61 to 0.74 in some seeds, but rule-off (`oc_anat_ruleoff_s2`) and dense (`oc_dense_s2_u600_t24`) runs also reached 0.74 in single seeds, and the structured-mask run gave the lowest test MSE for the unsigned fly mask. The working notes report the opposite conclusion for that mask experiment. This conflict is unresolved.

## Limitations
- The SRCN topology is synthetic, not a measured fly connectome. The docs say explicitly that it is not a fly reconstruction.
- SRCN supervised results use 3 seeds, one graph and one initialization draw per seed, a quick preset (200 steps), and no per-model learning-rate tuning. The docs list these as reasons the results are exploratory.
- Compute is measured, not matched. Parameter budgets differ across models, and the sparse models were not benchmarked against sparse execution kernels. The quick suite ran on CPU; the `cuda-smoke` run is a single-seed execution check, not a benchmark.
- Validation and test episodes use independent random streams, so overlaps can occur by chance; no disjoint split is claimed.
- The SRCN reward task is an immediate-reward bandit. It has no delayed credit assignment, and the docs note that several topologies can solve it.
- Flynet has no written report. Its learning-efficiency conclusions come from working notes; the learned criterion and the 41-seed count are not in the repository.
- Flynet odor-conditioning runs use different update budgets (`u600_t24`, `u2000_t24`, `u600_t48`, default) and mostly one or four seeds, so runs cannot be compared cleanly across rows.
- The structured-mask log contains repeated step blocks from more than one invocation, and the CSV holds one run per mask. It is unclear which invocation the CSV reflects.
- Flynet's plasticity and dopamine-gating rules are engineering abstractions, not validated biological models.

## Key files
- `srcn/srcn/experiment.py`: supervised comparison, presets (`quick`, `research`), checkpoints.
- `srcn/srcn/models.py`: SRCN and baseline recurrent models; sparse COO recurrence; state reset and carry-over.
- `srcn/srcn/graphs.py`: module-and-hub graph generation and degree-preserving rewiring.
- `srcn/srcn/tasks.py`: delayed recall, rule switching, transitive ordering, out-of-distribution and sensor-loss conditions.
- `srcn/srcn/reward.py`: unannounced-reversal reward task, action-value learner, learning-disabled control.
- `srcn/srcn/report.py`: regenerates `report.md` and `summary.csv` from saved runs.
- `srcn/tests/test_core.py`, `srcn/tests/test_reward.py`: degree preservation, sparse/dense equivalence, parameter budgets, task labels, reward recovery test.
- `srcn/results/baselines-quick/`: 54-run supervised suite (report, summary CSV, manifest, per-run JSON and checkpoints).
- `srcn/results/reward-online-q/`: action-value reward results.
- `srcn/results/reward-quick/`: earlier actor-critic pilot (did not recover).
- `srcn/results/cuda-smoke/`, `srcn/results/smoke-initial/`: single-seed execution checks; not benchmark results.
- `srcn/docs/research-notes.md`: literature context, game-demonstration context and next experiments.
- `flynet/flynet/connectome/graph.py`: MaleCNS loader, subgraph extraction, shuffled and random controls.
- `flynet/flynet/models/connectome_rnn.py`: masked RNN and actor-critic agent.
- `flynet/flynet/baselines/gru.py`: GRU baseline.
- `flynet/flynet/models/typed_rnn.py`: per-neuron-type shared dynamics (branch A).
- `flynet/flynet/models/plasticity.py`: three-factor plasticity at KC to MBON synapses; dopamine gating (branches B and C).
- `flynet/flynet/environments/odor_navigation.py`, `flynet/flynet/environments/odor_conditioning.py`: task environments.
- `flynet/flynet/experiments/learning_efficiency.py`, `odor_conditioning.py`, `lifetime_learning.py`, `structured_sparsity.py`: experiment entry points.
- `flynet/results/`: raw outputs (`le_*.csv/log`, `oc_*.csv/log`, `structured_sparsity.csv/log`, `run.log`). No summary report.
- `flynet/tests/`: tests for the connectome, models, environments, plasticity and structured sparsity.

## Sources
- `srcn/README.md`: task definitions, comparison models, run commands, reporting rules.
- `srcn/docs/research-notes.md`: scope and literature limits.
- `srcn/results/baselines-quick/report.md`, `summary.csv`, `manifest.json`: supervised numbers, paired differences, settings, platform.
- `srcn/results/reward-online-q/report.md`, `manifest.json`: online reward numbers and settings.
- `srcn/results/reward-quick/report.md`: actor-critic pilot numbers.
- `srcn/results/cuda-smoke/report.md`, `srcn/results/smoke-initial/report.md`: execution checks (cited for scope only).
- `srcn/tests/test_core.py`, `srcn/tests/test_reward.py`: task and test definitions.
- `flynet/README.md`: data source, experiment layout, agent definitions.
- `flynet/results/oc_*.log`, `flynet/results/oc_control.log`: odor-conditioning late-trial accuracies.
- `flynet/results/structured_sparsity.csv`, `flynet/results/structured_sparsity.log`: structured-mask final MSE and log structure.
- `flynet/results/le_*.csv`: learning-efficiency reward traces (not summarized into a table here; criterion undefined).
- `flynet/flynet/environments/odor_navigation.py`, `flynet/flynet/environments/odor_conditioning.py`, `flynet/flynet/experiments/structured_sparsity.py`: task definitions.
- `flynet/.remember/today-2026-09-12.md`: working notes, used only for the numbers and claims marked "from working notes" or "unverified".
