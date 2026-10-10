# Executive control of retrieval
**Dates:** 2026-09-10 · **Status:** paused · **Hardware/scale:** Local Ollama serving `qwen2.5:14b` (Q4_K_M), loopback only; GPU not recorded in the project docs. Two runs, 188 local generation calls in total (96 retrieval-gate development calls, 92 instruction-delivery calls). No training, no cloud inference.

## Question
Should the decision model itself choose when to consult memory, or do fixed and simple policies do as well at lower cost? The test uses a synthetic certificate-lookup task where some cases need a stored record and some have stale, expired, or conflicting evidence. A follow-up asked whether the instructions and evidence actually reach the model and are used.

## Approach
- **Model:** Qwen2.5 14B via local Ollama, temperature 0, seed 42. Frozen; no weights trained or changed.
- **Task:** Synthetic certificate lookups in two suites: *access* (one field) and *dispatch* (two fields). Eight evaluator-only scenarios per suite, including current-sufficient, memory-required, expired-current, unofficial-current, missing-everywhere, stale-only, conflict, and current-with-stale-distractors.
- **Memory:** Passive immutable store with exact entity-key search, limit 8, stable ID order. It has no relevance labels and does no validity filtering. Memory writes are out of scope.
- **Policies compared (retrieval run):** `no_retrieval`, `always_retrieve`, `heuristic` (retrieve unless every field has exactly one currently valid value), `model_gate` (the Qwen model chooses act or retrieve in one JSON call), `full_context` (the whole corpus is loaded), and `symbolic` (the heuristic gate plus a deterministic resolver, zero LLM calls).
- **Controls:** The symbolic arm as a deployable ceiling; full-context as an upper bound on information; identical answer prompt and JSON schema across LLM arms; randomized arm and task order; a hard per-episode budget (2 generation calls, 1 search, 32,768 tokens, 120 s).
- **Pre-registration:** `PROTOCOL.md` was written before any model call, with development gates, a useful-effect threshold, and a stopping rule. The run copied it with hashes. Development stopped at the readiness gate, so the 48 held-out cases were not run.
- **Delivery follow-up:** `INSTRUCTION_DELIVERY_PROTOCOL.md` (v2) was written before inference. Stage A tested three delivery routes (`chat_system`, `chat_user`, `raw_system`) and a no-instruction negative control on 8 nonce tasks each. Stage B gave 20 development tasks with retrieval always provided, so no gate decision was tested. Stage C (fresh confirmation) was conditional on a route qualifying, and none did.

## Results

| Test | Result | Baseline / control | Verdict |
| --- | --- | --- | --- |
| Retrieval run, access suite (8 dev cases): correct | no_retrieval 5/8; always_retrieve 6/8; heuristic 6/8; model_gate 6/8; full_context 4/8 | symbolic 8/8 | Readiness failed: retrieval lift 1/8 (rule needs ≥2); full_context below 6/8 |
| Retrieval run, dispatch suite (8 dev cases): correct | no_retrieval 5/8; always_retrieve 5/8; heuristic 6/8; model_gate 6/8; full_context 5/8 | symbolic 8/8 | Readiness failed: retrieval lift 0/8 |
| Searches, both suites | model_gate 10 (5 + 5); heuristic 12 (6 + 6); always_retrieve 16 (8 + 8); full_context 0 (loads all records) | symbolic 12 (6 + 6) | Model gate used 2 fewer searches than heuristic and 6 fewer than always_retrieve; both avoided searches were missed retrievals (see next row) |
| Missed needed retrieval (3 needs-memory cases per suite) | model_gate 1/3 (access) and 1/3 (dispatch); always_retrieve, heuristic, symbolic 0/3 each | — | Negative: the model gate skipped retrieval on expired-current cases |
| Total prompt + output tokens, both suites | model_gate 13,439 (gate calls alone 6,202); always_retrieve 8,185; heuristic 7,631; full_context 26,275 | — | Negative: model gate used 64% more tokens than always_retrieve and 76% more than heuristic |
| Stale-evidence use | full_context 3 (access 2, dispatch 1) | model_gate, always_retrieve, heuristic, symbolic 0 | Negative for full_context |
| Heuristic ceiling rule | Not triggered (heuristic 6/8 in both suites, not 8/8) | symbolic 8/8 | Not met; suite not retired as cheaply resolved |
| Run integrity | 96 generation calls; 0 API failures; 0 invalid-JSON episodes; 7 preflight tests and all post-run audit checks passed | — | Pass |
| Delivery, Stage A nonce controls (8 tasks per route) | chat_system 8/8; chat_user 8/8; raw_system 8/8; chat_none (expects UNKNOWN) 8/8 | Pass threshold 7/8 | Pass: simple instructions reached the model on all three routes |
| Delivery, Stage B access (10 dev tasks, fixed retrieval): correct (evidence-valid; stale use) | chat_system 6/10 (8/10; 2); chat_user 6/10 (9/10; 1); raw_system 7/10 (7/10; 2) | symbolic 10/10 | Fail: no route reached 9/10 |
| Delivery, Stage B dispatch (10 dev tasks, fixed retrieval): correct (evidence-valid; stale use) | chat_system 8/10 (9/10; 1); chat_user 7/10 (8/10; 2); raw_system 8/10 (9/10; 1) | symbolic 10/10 | Fail: no route reached 9/10 in both suites with zero stale use |
| Delivery, Stage C confirmation | Not run (no candidate route) | — | Not run |
| Delivery run totals | 92 generation calls; 35,887 prompt + output tokens; 0 infrastructure failures | — | Descriptive |

## What it showed
- The model-controlled gate did not beat simple code. It matched the heuristic at 6/8 in both suites, used fewer searches, but missed necessary retrieval on two expired-current cases and used 76% more tokens than the heuristic.
- Retrieval itself barely helped on these tasks: always retrieving beat never retrieving by 1 case on access and 0 on dispatch.
- The fully symbolic baseline solved both suites (8/8) with no LLM calls. These structured tasks do not need a language model, so the LLM-gate comparison does not test a case where executive control is required.
- Simple instructions reached the model on all three delivery routes (8/8 each), but evidence use stayed unreliable on the 20 development tasks. The best routes reached 8/10 on dispatch and 6–7/10 on access, and every route had stale-evidence uses. The answer step, not the retrieval decision, was the bottleneck.
- Overall the first pilot is a negative result. The architectural hypothesis (the decision model owns recall) remains untested in a setting where a simple resolver is insufficient.

## Limitations
- Synthetic, template-generated task. Only 8 development cases per suite in the retrieval run and 10 per suite in the delivery run; one observation per arm. Differences of one or two cases are not meaningful.
- Held-out evaluation (48 cases) was never run, and the delivery confirmation stage was never run.
- One prompted, frozen model (Qwen2.5 14B, Q4_K_M). No trained gate, no other models, no model-written queries, no memory writes.
- The retrieval query was fixed to exact entity matching, so the test covers only the binary retrieve-or-act choice.
- Evidence-valid counts measure citations, not correct reasoning. Private reasoning was not observed, so failure causes are inferred from outputs only.
- Latency includes model loading and host caching. GPU determinism is not guaranteed, though seeds and temperature were fixed.
- The earlier MemRL pilot is closed and inconclusive; no MemRL data were used here.

## Key files
- `README.md`: component boundary, both runs, and reproduction commands.
- `DESIGN.md`: the broader hypothesis that the decision model owns retrieval intent.
- `PROTOCOL.md`: pre-registered retrieve-versus-act protocol (arms, budgets, development gates, stopping rule).
- `PRIOR_WORK.md`: review of CoALA, Adaptive-RAG, Self-RAG, and Rational Metareasoning.
- `MODEL.md`: specification of the intended learned executive policy; states that no trained weights exist.
- `INSTRUCTION_DELIVERY_PROTOCOL.md`: pre-registered v2 protocol for the instruction-delivery and evidence-use follow-up.
- `experiment.py`: passive store, heuristic and symbolic policies, runner with budget enforcement, scoring, development gates.
- `instruction_delivery.py`: delivery routes, request planning, client, and candidate selection for the follow-up.
- `analyze.py`, `analyze_delivery.py`: offline audits of saved runs (no model calls).
- `test_experiment.py`, `test_instruction_delivery.py`: preflight unit tests.
- `CONCLUSION.md`: research decision after the retrieval run.
- `results/retrieve_act_20260910T224325010154Z/`: retrieval run directory (REPORT, summary, gates, audit, traces).
- `results/instruction_delivery_20260910T225955352340Z/`: delivery run directory (REPORT, summary, audit, traces).

## Sources
- `README.md`
- `DESIGN.md`
- `PROTOCOL.md`
- `PRIOR_WORK.md`
- `MODEL.md`
- `CONCLUSION.md`
- `INSTRUCTION_DELIVERY_PROTOCOL.md`
- `results/retrieve_act_20260910T224325010154Z/REPORT.md`
- `results/retrieve_act_20260910T224325010154Z/summary.json` (per-arm counts, searches, tokens; used to check the totals)
- `results/retrieve_act_20260910T224325010154Z/model_metadata.json`
- `results/instruction_delivery_20260910T225955352340Z/REPORT.md`
- `results/instruction_delivery_20260910T225955352340Z/summary.json`
