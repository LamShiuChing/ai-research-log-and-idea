# Retrieve-versus-act candidate experiment — 2026-09-10

Status: **stopped_after_development**. This is a development diagnostic, not a benchmark reproduction or evidence of general executive control.

The frozen protocol and code preceded the first local generation. Memory remained a passive immutable store; the model gate chose act/retrieve and the host executed the fixed entity query. No training or memory writes occurred.

Run: 96 local Qwen2.5 14B generation calls, 61,024 actual prompt/output tokens, 48.4 seconds elapsed. Zero cloud inference calls. Model digest and Ollama metadata are in `model_metadata.json`.

## Results

Counts are per 8-case development suite. Token totals include gate and answer calls. Full-context uses no search call but loads the entire corpus; its memory and token cost is included. See [interpretation](../../CONCLUSION.md) for the resulting research decision.

| Suite | Policy | Correct | Searches | Prompt tokens | Output tokens | Wall seconds | Stale use |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| access | no_retrieval | 5/8 | 0 | 2,364 | 144 | 2.504 | 0 |
| access | always_retrieve | 6/8 | 8 | 3,422 | 150 | 2.825 | 0 |
| access | heuristic | 6/8 | 6 | 3,186 | 152 | 2.569 | 0 |
| access | model_gate | 6/8 | 5 | 5,857 | 215 | 7.671 | 0 |
| access | full_context | 4/8 | 0 | 10,732 | 196 | 4.257 | 2 |
| access | symbolic | 8/8 | 6 | 0 | 0 | 0.000 | 0 |
| dispatch | no_retrieval | 5/8 | 0 | 2,779 | 207 | 3.909 | 0 |
| dispatch | always_retrieve | 5/8 | 8 | 4,317 | 296 | 4.978 | 0 |
| dispatch | heuristic | 6/8 | 6 | 3,993 | 300 | 5.065 | 0 |
| dispatch | model_gate | 6/8 | 5 | 7,004 | 363 | 7.122 | 0 |
| dispatch | full_context | 5/8 | 0 | 14,987 | 360 | 7.410 | 1 |
| dispatch | symbolic | 8/8 | 6 | 0 | 0 | 0.000 | 0 |

## Protocol decision

- **access: readiness failed.** Always-retrieve 6/8; no-retrieval 5/8; full-context 4/8; heuristic 6/8; symbolic 8/8. Proceed to held-out evaluation: False.
- **dispatch: readiness failed.** Always-retrieve 5/8; no-retrieval 5/8; full-context 5/8; heuristic 6/8; symbolic 8/8. Proceed to held-out evaluation: False.

## Retrieval decisions and evidence

| Suite | Policy | Missed needed retrieval | Search on sufficient current evidence | Search on missing/stale/conflict | Gate tokens | Evidence-valid outputs |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| access | always_retrieve | 0/3 | 2/2 | 3 | 0 | 8/8 |
| access | heuristic | 0/3 | 0/2 | 3 | 0 | 8/8 |
| access | model_gate | 1/3 | 0/2 | 3 | 2,896 | 8/8 |
| access | full_context | 0/3 | 0/2 | 0 | 0 | 6/8 |
| access | symbolic | 0/3 | 0/2 | 3 | 0 | 8/8 |
| dispatch | always_retrieve | 0/3 | 2/2 | 3 | 0 | 7/8 |
| dispatch | heuristic | 0/3 | 0/2 | 3 | 0 | 8/8 |
| dispatch | model_gate | 1/3 | 0/2 | 3 | 3,306 | 8/8 |
| dispatch | full_context | 0/3 | 0/2 | 0 | 0 | 7/8 |
| dispatch | symbolic | 0/3 | 0/2 | 3 | 0 | 8/8 |

Evidence-valid counts measure citation coverage/validity; they are not success counts and do not establish that conflicting evidence was resolved correctly. An incorrect abstention can still have the required empty evidence list. Full-context's missed retrieval count is zero because it receives all storage evidence.

| Suite | Policy | Loaded records (sum) | Memory bytes (sum) | Search/read microseconds (sum) | Gate seconds (sum) |
| --- | --- | ---: | ---: | ---: | ---: |
| access | no_retrieval | 0 | 0 | 1.3 | 0.000 |
| access | always_retrieve | 13 | 2,011 | 97.0 | 0.000 |
| access | heuristic | 10 | 1,549 | 107.0 | 0.000 |
| access | model_gate | 8 | 1,240 | 63.3 | 4.901 |
| access | full_context | 104 | 16,040 | 229.0 | 0.000 |
| access | symbolic | 10 | 1,549 | 71.7 | 0.000 |
| dispatch | no_retrieval | 0 | 0 | 1.3 | 0.000 |
| dispatch | always_retrieve | 19 | 2,921 | 96.5 | 0.000 |
| dispatch | heuristic | 15 | 2,308 | 59.5 | 0.000 |
| dispatch | model_gate | 12 | 1,847 | 65.8 | 1.908 |
| dispatch | full_context | 152 | 23,320 | 288.5 | 0.000 |
| dispatch | symbolic | 15 | 2,308 | 78.6 | 0.000 |

Search/read time measures the local loading branch (including minimal bookkeeping even when no records are loaded); it is not a remote search price. All 80 LLM episodes produced valid schema actions with zero infrastructure failures. Exact matching requests agreed on parsed JSON actions; observed formatting differences did not change actions.

## Observed failures

- `p91001/2` (expired_current): gate `act`, answer `ABSTAIN`, expected `1527`. Inspect the complete observable request/response in `generation_calls.jsonl`.
- `p91001/3` (unofficial_current): gate `retrieve`, answer `ABSTAIN`, expected `1330`. Inspect the complete observable request/response in `generation_calls.jsonl`.
- `p91002/2` (expired_current): gate `act`, answer `ABSTAIN`, expected `4120/4919`. Inspect the complete observable request/response in `generation_calls.jsonl`.
- `p91002/6` (conflict): gate `retrieve`, answer `4371/3299`, expected `ABSTAIN`. Inspect the complete observable request/response in `generation_calls.jsonl`.

## Scope and reproduction

The two suites share certificate validity rules and a synthetic task family. Eight cases per suite cannot establish generalization or noninferiority. Frozen seeds do not guarantee bitwise GPU determinism. Latency includes model loading and depends on Ollama caching and host load; symbolic times near zero are CPU measurements, not literal zero cost.

All evaluator scenarios/targets are separate from model-facing observations. An offline audit checked snapshot hashes, exact prompts/search results, observable action traces, recomputed scoring, token reconciliation, budgets, and the development stopping rules. See `audit.json`. Seven preflight unit tests passed before the run; their source is saved beside the protocol.

Reproduce from `D:\research\executive-control`: `python -m unittest -v test_experiment.py`, then `python experiment.py --run`. Each run creates a new directory and applies its own development stop. Analyze a saved run with `python analyze.py results/<run-name>`; this makes no model calls.

Preserved artifacts: `PROTOCOL.md`, `PRIOR_WORK.md`, `experiment.py`, `test_experiment.py`, `prompts.json`, `manifest.json`, `model_metadata.json`, `dataset_evaluator_only.json`, `generation_calls.jsonl`, `episodes.jsonl`, `development_gates.json`, `summary.json`, `paired_comparisons.json`, and `audit.json`. The original MemRL pilot remains closed as inconclusive and unchanged.
