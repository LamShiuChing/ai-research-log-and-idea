# Instruction-delivery diagnostic — 2026-09-10

Status: **stopped_after_development**. Development candidates: []. Confirmation executed: False.

All instruction controls passed. This run does not support the hypothesis that chat system instructions are simply being dropped. It does not establish a universally reliable transport or localize an internal server cause.

Executed 92 local Qwen2.5 14B generation calls, 35,887 prompt-plus-output tokens, 48.7 seconds. Infrastructure failures: 0. No training or cloud inference. The protocol includes five preflight tests; all 17 offline audit checks passed.

## Controls

| Delivery route | Correct marker / expected UNKNOWN | Prompt tokens | Output tokens | Wall seconds |
| --- | ---: | ---: | ---: | ---: |
| chat_system | 8/8 | 519 | 167 | 5.814 |
| chat_user | 8/8 | 647 | 167 | 2.750 |
| raw_system | 8/8 | 519 | 167 | 2.629 |
| chat_none | 8/8 | 432 | 56 | 1.033 |

The marker was only in the instruction, never in the output schema or no-instruction payload. The negative control expects UNKNOWN. Successful positive controls establish instruction-conditioned behavior on these simple tasks.

## Evidence-use development

Every route received identical current evidence and mechanical search results, including distractors. The semantic instruction and JSON schema were unchanged across routes. Both code alphabet and task identities are fresh, so differences versus the old experiment are not a controlled delivery effect.

| Suite | Route | Correct | Evidence-valid | Stale use | Prompt tokens | Cached prompt subset | Output tokens | Wall seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| access | chat_system | 6/10 | 8/10 | 2 | 4,473 | 3972 | 232 | 3.937247 |
| access | chat_user | 6/10 | 9/10 | 1 | 4,633 | 2049 | 266 | 5.060564 |
| access | raw_system | 7/10 | 7/10 | 2 | 4,473 | 2376 | 266 | 4.826644 |
| access | symbolic | 10/10 | 10/10 | 0 | 0 | n/a | 0 | 0.000742 |
| dispatch | chat_system | 8/10 | 9/10 | 1 | 5,807 | 3529 | 407 | 7.042899 |
| dispatch | chat_user | 7/10 | 8/10 | 2 | 5,967 | 2249 | 475 | 8.572714 |
| dispatch | raw_system | 8/10 | 9/10 | 1 | 5,807 | 4360 | 407 | 6.981997 |
| dispatch | symbolic | 10/10 | 10/10 | 0 | 0 | n/a | 0 | 0.000534 |

Cached prompt counts are part of total prompt counts, not additional tokens. Cache reuse differs across routes; latency is descriptive and includes model loading. Evidence-valid citations alone do not prove the right answer or correct conflict handling.

No route reached 9/10 in each suite with zero stale use. Do not select a route merely because it has the highest aggregate score. All 20 new confirmation tasks remain unrun, along with the original study's 48 held-out tasks.

## Observable failures

| Task | Scenario | Route | Actual answer | Expected |
| --- | --- | --- | --- | --- |
| v2/101101/access/3 | unofficial_current | chat_user | ABSTAIN | CE07CC2 |
| v2/101101/access/3 | unofficial_current | chat_system | ABSTAIN | CE07CC2 |
| v2/101101/access/5 | wrong_entity_current | chat_user | ABSTAIN | C378B7D |
| v2/101101/access/5 | wrong_entity_current | raw_system | CE11B5D | C378B7D |
| v2/101101/access/5 | wrong_entity_current | chat_system | ABSTAIN | C378B7D |
| v2/101101/access/4 | future_current | raw_system | CDA8593 | CE54542 |
| v2/101101/access/4 | future_current | chat_system | CDA8593 | CE54542 |
| v2/101101/access/4 | future_current | chat_user | CDA8593 | CE54542 |
| v2/101101/access/2 | expired_current | raw_system | C0E8838 | C2BD9B2 |
| v2/101101/access/2 | expired_current | chat_system | C0E8838 | C2BD9B2 |
| v2/101101/access/2 | expired_current | chat_user | ABSTAIN | C2BD9B2 |
| v2/101101/dispatch/2 | expired_current | raw_system | ABSTAIN | C969E08/CBD1C0C |
| v2/101101/dispatch/2 | expired_current | chat_user | ABSTAIN | C969E08/CBD1C0C |
| v2/101101/dispatch/2 | expired_current | chat_system | ABSTAIN | C969E08/CBD1C0C |
| v2/101101/dispatch/4 | future_current | chat_system | CA85AE2/CF99008 | C173976/C6A869D |
| v2/101101/dispatch/4 | future_current | chat_user | CA85AE2/CF99008 | C173976/C6A869D |
| v2/101101/dispatch/4 | future_current | raw_system | CA85AE2/CF99008 | C173976/C6A869D |
| v2/101101/dispatch/9 | current_stale | chat_user | C5D439D/C26F28F | C5D439D/CDC5E01 |

Inspect the scenario breakdown above for the evidence judgments that failed. Retrieval intention was held fixed, so these errors cannot be explained by failure to issue a search. The traces do not reveal private reasoning or prove why an instruction was misapplied.

## Research decision

Keep the existing chat transport as an unvalidated baseline; neither change the service nor claim an interface repair. The next prerequisite is a fresh, minimal-pair evidence-use diagnostic that isolates entity matching, authority, time validity, conflict handling, and their composition. That should be an answer-policy capability check, not another retrieval-gate contest. Keep the certificate task as a symbolic regression fixture and do not train an executive gate to compensate for its answer-policy failures.

The intended learned executive policy, observations, actions, and learning target are specified in [MODEL.md](../../MODEL.md). This run trains no executive model and supplies no evidence of an advantage over simple code.

## Reproduction and preserved evidence

From `D:\research\executive-control`: `python -m unittest -v test_instruction_delivery.py`, then `python instruction_delivery.py --run`. Analyze a saved directory with `python analyze_delivery.py results/<run-name>`. The audit makes no inference calls.

The run stores the protocol, source snapshots, dataset, every planned request, every attempted request/response, model template and version, summaries, paired comparisons, gate decisions, and audit. All preceding result directories and MemRL files are preserved. Small samples and shared synthetic templates limit generalization; paired comparisons are descriptive.
