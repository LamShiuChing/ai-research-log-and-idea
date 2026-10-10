# ARC-AGI-3 interactive-game agents
**Dates:** 2026-09-25 (theory and first baseline run) to 2026-10-02 (explorer reruns) · **Status:** Draft. Baselines run. The v0.3 agent is not built, and its kill-line tests have not been run. · **Hardware/scale:** No hardware spec recorded. Torch is pinned to a CUDA build, and the F8-lite probe uses device `cuda`. Scale: 25 public games, 2,000 to 5,000 actions per game.

## Question
Can an agent that never predicts the screen, and instead learns which actions change anything, where changes happen, what it has already tried in similar situations, and what winning looked like, clear levels in interactive games more efficiently than a graph-based explorer? The v0.3 theory states this as a falsifiable claim.

## Approach
- **Benchmark:** ARC-AGI-3 public games (25 locally cached). The agent gets a grid of colour values (0 to 15), the list of changed cells since the last step, level-complete and game-over signals, and the available actions. Each action spends budget. `run_eval.py` defaults to 5,000 actions per game.
- **Scoring:** Per level, `min((baseline / actions)^2, 1.15)`, averaged with weights by level number. Unfinished levels score 0. This is a local re-implementation of the official scorer, not the official one.
- **Splits (THEORY.md §6):** Design (13 games) and held-out (12 games), assigned by sha256 of the game-id prefix.
- **Agent versions:**
  - v0.2, "predict the next screen, then plan": tested only as prediction probes (F1, F2, F2b, F8-lite). Killed per THEORY.md L1. No agent was built and scored.
  - v0.3, "Little Scientist" (decision first, no vision): four learned claims (clock-key removal, focus on change, situation counts, win contrast) plus exact-memory walking. Theory only; no code in `arc3/agents.py`.
- **Baselines:** `RandomAgent` and `GraphExplorer` (registered as `random` and `explorer`).
- **Pre-registered kill lines (THEORY.md §6, stated as written before building):**
  - K1 (design games, level 1): pass at 9/13 or more and a median of 185 actions or fewer; kill at 7/13 or fewer and a median above 276.
  - K2 (click design games): pass at 4/5 and a median of 122 or fewer; kill if median is above 183.
  - K3 (levels 2 and up, actions divided by human baseline): pass at 3 or less; kill above 10.
  - M1 (held-out score): continue at 1% or more; stop the decision-first line below 0.1%.

## Results

| Agent | Score | Baseline | Verdict |
|---|---|---|---|
| Graph explorer (`explorer`, budget 5000) | 0.04% (harness text, three identical reruns) | none (baseline) | Baseline. Level 1 cleared in 7 of 13 design games, median 553 actions (checked against the JSON). |
| Random (`random`, budget 2000) | 0.0004565853892325694 (JSON total, fraction) | explorer 0.04% | Baseline. Smaller budget, so not directly comparable. |
| v0.2 predict-next-screen | not run (as an agent) | n/a | Killed by probes. Prediction accuracy only: exact laws median 0.9% (F2b), small net median 1.6% (F8-lite). These are not game scores. |
| v0.3 Little Scientist (decision first) | not run | explorer 0.04% | Not run. Kill lines K1, K2, K3 and M1 are untested. |

## What it showed
- The graph explorer scored 0.04% over the 25 public games at 5,000 actions per game. It cleared at least one level in 13 of the 25 games.
- Random play at 2,000 actions scored slightly higher in total than the explorer (0.0004565853892325694 against 0.0003926909377569799), though it cleared levels in only 3 games. The random run used a smaller budget, so this does not show random play beats the explorer. It does show that the aggregate score is very small and is driven by a few levels.
- The v0.2 idea of predicting the whole next screen did not work in the probes. Exact rules and a small network each reached only about 1% median accuracy, and they failed in different games.
- v0.3 is a theory with explicit claims and kill lines. None of its claims has been tested in code, so there is no evidence yet for or against it.

## Limitations
- The scorer is a local re-implementation of the official ARC-AGI-3 scoring, not the official scorer.
- All 25 games are public. The THEORY.md notes that hidden games are deliberately different, so public results may not transfer. The held-out split is also public games, not a private test set.
- The v0.3 agent does not exist. The M1 held-out score has not been measured.
- The random and explorer runs used different budgets (2,000 and 5,000).
- An earlier explorer run (`results/explorer-1790358184.json`, total fraction 1.0287802797219008e-05, 3 games cleared) differs from the later explorer runs (total fraction 0.0003926909377569799, 13 games cleared). The code version of the earlier run is not recorded, so the difference is not explained.
- The probe figures (F2b 0.9%, F8-lite 1.6%) come from THEORY.md and commit 0ae5d5e. Their output files are not in `results/`, so they were not re-checked here.
- The design-split level-1 counts (7 of 13) and median (553) were re-computed from the JSON and match THEORY.md §6.

## Key files
- `THEORY.md`: v0.3 theory, claims D1 to D6, input contract, kill lines K1 to K3 and M1, open problems, and the results table.
- `run_eval.py`: evaluation harness. Runs one agent on all cached games under an action budget, prints per-game scores, and writes `results/<agent>-<timestamp>.json`.
- `arc3/agents.py`: `RandomAgent` and `GraphExplorer` baselines and the `AGENTS` registry (`random`, `explorer`). No v0.3 agent.
- `arc3/scoring.py`: level score and weighted game score.
- `arc3/lens.py`, `arc3/laws.py`: helpers imported by the v0.2 probes (event and explanation helpers, and exact-law induction).
- `probes/f1_object_events.py`: F1, share of changed cells explained by whole-object events. Defines the design split and step settings.
- `probes/f2_laws.py`: F2, whether exact object laws from one trajectory half predict the other half.
- `probes/f8_lite.py`: F8-lite, a small per-game convolutional network on the F2b data, device `cuda`.
- `probes/hidden_state.py`: checks whether identical frame plus identical action can give different next frames.
- `probes/pixel_locality.py`: lookup-table probe of local predictability of cell values.
- `probes/testbed_facts.py`: determinism, change sparsity, no-op rate, and frames per step.
- `results/explorer-*.json`, `results/random-*.json`: per-game run outputs. `results/explorer_rerun_1.txt` to `_3.txt`: console output of three reruns.
- `pyproject.toml`: dependencies (arc-agi, numpy, scipy, torch).

## Sources
- `THEORY.md` in the project folder (v0.3 theory; v0.2 evidence table; commits 884d512 for v0.3 and 0ae5d5e for v0.2 results).
- `results/` JSON and text outputs (dated 2026-09-25 and 2026-10-02 by file and log timestamps).
- ARC Prize methodology (scoring, actions-only): https://docs.arcprize.org/methodology.md
- Hidden-game note cited in THEORY.md: https://arxiv.org/html/2603.24621v1
- Graph-Based Exploration for ARC-AGI-3 (prior art, cited in THEORY.md): https://arxiv.org/html/2512.24156v1
