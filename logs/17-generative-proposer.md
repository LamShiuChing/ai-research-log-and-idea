# Generative proposer: can a CVAE design Boolean circuits for unseen truth tables?
**Dates:** Sep 19, 2026 (validation run) · **Status:** One validation run, single seed; test split not yet evaluated; no advantage over baseline shown · **Hardware/scale:** NVIDIA GeForce RTX 4080 SUPER; PyTorch 2.11.0+cu128; 96,008 parameters; training 92.20 s

## Question
Can a small conditional variational autoencoder (CVAE) propose Boolean circuits that match truth-table goals absent from its training data, and does it do better than retrieval plus random mutation?

## Approach
- **Model:** CVAE with a context encoder (two hidden layers, width 128), a conditional prior and training-only posterior over a 16-dimensional Gaussian latent, and a categorical decoder (two hidden layers, width 128). 96,008 trainable parameters. Legal-source and commutativity masks enforce syntax.
- **Domain:** circuits of four gates (AND, OR, XOR, NOT) over three inputs. The goal is an 8-bit truth table. Correctness is checked exactly on all eight inputs.
- **Data:** 100,000 random circuit samples (seed 17), deduplicated by rooted expression: 1,629 distinct expressions over 61 eligible goal truth tables. Goals are split disjointly: 42 train / 9 validation / 10 test. Records: 1,119 train / 200 validation / 310 test. Up to 64 demonstrations per goal, minimum 12.
- **Training:** AdamW, learning rate 0.002, batch size 256, 1,000 steps, seed 23. KL weight warmed up to 0.05 over the first 40% of steps.
- **Baselines:**
  - Random: grammar-valid random circuits.
  - Retrieval + mutation: take the training circuits whose goal is nearest in Hamming distance, then apply one (75%) or two (25%) random gate replacements.
  - CVAE with latent noise (CVAE).
  - The same CVAE with latent noise off, categorical output sampling still on (CVAE, prior mean). This is a diagnostic, not a separately trained model.
- **Proposals:** 64 per goal for each method, on the 9 validation goals (576 per method).

## Results
Validation split, seed 23, 64 candidates per goal. "Correct proposals" = candidates that match the goal on all eight rows.

| Method | Goals solved | Correct proposals | Other |
|---|---:|---:|---|
| Random | 0 / 9 | 0 / 576 (0.0%) | 0.00 distinct correct per goal; 6.01 ms proposal + evaluation |
| Retrieval + mutation | 4 / 9 | 7 / 576 (1.2%) | 0.78 distinct correct per goal; 4.69 ms |
| CVAE | 4 / 9 | 5 / 576 (0.9%) | 0.44 distinct correct per goal; 24.87 ms |
| CVAE, prior mean | 1 / 9 | 2 / 576 (0.3%) | 0.11 distinct correct per goal; 9.26 ms |

All methods produced 0.0% invalid proposals. Times are one batched measurement on this machine, not stable latency benchmarks; they exclude model initialization and data generation.

## What it showed
- The pipeline runs end to end, and every proposal is syntactically valid by construction. Behavioral correctness is low: at most 1.2% of candidates are correct.
- The CVAE solved as many validation goals as retrieval plus mutation (4 of 9), but found fewer correct proposals (5 vs 7) and fewer distinct correct designs per goal (0.44 vs 0.78). It also took longer in this measurement (24.87 ms vs 4.69 ms).
- Training reconstruction loss fell from 15.29 at step 1 to 0.19 at step 1000. Low training loss did not translate into correct designs on held-out goals, so reconstruction alone is not the success criterion.
- Turning off latent noise reduced solved goals from 4 to 1. That hints the latent helps in this run, but one seed and nine goals cannot establish it.
- The result is a narrow feasibility observation: the neural model emitted correct circuits for some unseen validation goals. It does not show a latent advantage, and retrieval plus mutation was the stronger method here.

## Limitations
- One training seed (23) and one 64-candidate evaluation per goal on nine validation goals. Results are exploratory.
- The test split (10 goals, 310 records) has not been used for model evaluation or tuning.
- No separately trained, matched non-latent or autoregressive baseline. The prior-mean row is a diagnostic only.
- Equal candidate counts are not equal compute budgets. Timings are single batched measurements, not latency benchmarks.
- Fixed 3-input, 4-gate domain. "Novelty" means distinct rooted expression structure, not scientific invention.
- The dataset is a capped random sample, not exhaustive, and it is not uniform over valid circuit structures.
- Canonicalization is not full Boolean simplification.
- CUDA runs are not bitwise reproducible.
- Retrieval plus mutation can see training circuits; the neural model sees only the goal bits at inference.

## Key files
- `README.md`: project overview, run commands, comparisons, and the next research gate.
- `research/THESIS.md`: working hypotheses (composition, representation, experience organization).
- `research/EXPERIMENT_001.md`: experiment specification written before the validation result: domain, data, model, evaluation metrics, and limits.
- `research/INITIAL_RESULTS.md`: write-up of the validation run, interpretation, next experiments, and test-split status.
- `gmm/model.py`: `ConditionalProposer`, the CVAE (context encoder, prior, posterior, decoder, loss, sampling).
- `gmm/experiment.py`: `run_experiment`, which trains the model, builds the baselines, and computes per-method metrics and the report.
- `gmm/circuits.py`: circuit representation, exact truth-table evaluation, random circuit generation, and `mutate`.
- `gmm/dataset.py`: dataset generation, canonical deduplication, and the disjoint goal split.
- `artifacts/data/circuits-v1/manifest.json`: goal masks per split, record counts, and file hashes.
- `artifacts/runs/validation-seed23/summary.json`: per-method metrics, training time, and limits.
- `artifacts/runs/validation-seed23/report.md`: the run's summary table.
- `artifacts/runs/validation-seed23/training.json`: training loss history.
- `artifacts/runs/validation-seed23/*_details.json`: per-goal results and example circuits.
- `tests/`: unit tests for circuits, dataset, metrics, and model (10 tests, reported as passing in `INITIAL_RESULTS.md`; not re-run for this log).

## Sources
- `research/INITIAL_RESULTS.md` (run description, results table, interpretation, next steps)
- `artifacts/runs/validation-seed23/report.md`, `summary.json`, `training.json`, `config.json`
- `artifacts/data/circuits-v1/manifest.json`
- `research/EXPERIMENT_001.md`, `research/THESIS.md`, `README.md`
- Planned follow-up, not built: the related "mind model" project's `PROJECT_STATE.md` is plan-only, with nothing built or measured yet.
