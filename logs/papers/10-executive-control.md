# Should a Local Model Decide When to Recall? A Negative Pilot on Retrieve-versus-Act Control

**Author:** Eric Lam (independent researcher, Toronto)

## Abstract

We ask whether a frozen local language model, Qwen2.5 14B, can decide when to consult a passive memory store, and whether that decision preserves task success while reducing computation compared with fixed and simple observable policies. The task is a synthetic certificate-lookup problem with current, expired, future, unofficial, conflicting, and missing evidence in two suites (one-field and two-field). Six policies were compared on 16 development cases: no retrieval, always retrieve, a simple observable heuristic gate, a model-controlled gate, full-context loading, and a deterministic symbolic resolver. The model gate matched the heuristic's accuracy (6/8 in each suite), made fewer searches than always retrieving (10 versus 16 across both suites), missed necessary retrieval twice, and used 13,439 tokens versus 7,631 for the heuristic. The pre-registered readiness rule failed in both suites, so no held-out evaluation was run. A follow-up found that simple instructions reached the model on all three delivery routes, but evidence use stayed unreliable (at most 8/10 per suite) and no route was confirmed. The fully symbolic baseline solved both suites without any model calls. We conclude that this pilot does not demonstrate useful executive control of retrieval, and that its task is too simple to test it.

## Introduction

A common architecture separates a passive memory store from the agent that decides what to remember, when to recall, and how to use recalled evidence. Prior work already treats retrieval as an action chosen by the language model: CoALA organizes agents around memory and internal and external actions (Sumers et al.), Adaptive-RAG trains a smaller model to choose between no, single-step, and iterative retrieval (Jeong et al.), Self-RAG trains one model to retrieve on demand (Asai et al.), and Rational Metareasoning learns when to spend computation with a value-of-computation reward (De Sabbata et al.). Our contribution is narrower. We hold the retrieval mechanism fixed, use a prompted and frozen model with no training, and compare the model's retrieve-or-act decision with fixed and simple observable policies under complete cost accounting.

The research question is: can the same local model choose when to consult memory while preserving task success and reducing unnecessary retrieval, compared with simple policies? We also tested whether the model reliably receives and uses the instructions and evidence that such a controller would depend on.

## Method

**Model and interface.** Qwen2.5 14B (Q4_K_M) was served by a local Ollama instance on a loopback endpoint, with temperature 0 and seed 42. Weights were not changed and no cloud inference was used. The decision model returned a JSON action (`act` or `retrieve`). On retrieval, the host executed a fixed query (exact entity equality, limit 8, stable record-ID order). The same model then gave a JSON answer or `ABSTAIN` from the supplied evidence. Memory was a passive immutable store with no relevance labels and no validity filtering.

**Tasks.** Each task asks for one field (access suite) or two fields (dispatch suite) for a fresh random entity, with an as-of date and four-digit or hexadecimal opaque codes. Records carry identifiers, source, issue and expiry dates, field, entity, and value. Official registry records are usable only while valid. Conflicting valid values require abstention. Each suite has eight evaluator-only scenarios: current sufficient; memory required; expired current replaced by valid memory; unofficial current corrected by memory; missing everywhere; stale memory only; unresolved valid conflict; and current sufficient with stale distractors. The model never saw the scenario label or expected answer.

**Policies.** (1) `no_retrieval`: one answer call on current evidence. (2) `always_retrieve`: fixed search, then one answer call. (3) `heuristic`: retrieve unless every requested field has exactly one distinct currently valid value. (4) `model_gate`: the model chooses act or retrieve, then answers. (5) `full_context`: the entire fixed corpus, including distractors, is loaded. (6) `symbolic`: the heuristic gate, search, and a deterministic validity and uniqueness resolver, with zero LLM calls.

**Budgets and pre-registration.** Each episode had at most two generation calls, one search, 32,768 prompt-plus-output tokens, and 120 seconds. The protocol was written and frozen before any model call. Development gates required symbolic 8/8, at least 95% valid model episodes, always-retrieve and full-context at least 6/8, and at least a 2/8 lift of always-retrieve over no-retrieval. A suite was retired if the heuristic was 8/8 with at least 25% fewer searches than always-retrieve.

**Delivery follow-up.** A second, separately pre-registered protocol tested whether the same instructions reach the model through three routes: a system chat message, a user message, and a raw ChatML system block. Stage A used eight nonce-marker tasks per route, with a no-instruction negative control. Stage B used 20 fresh certificate tasks with fixed retrieval, so the retrieval decision was not tested. A route could advance only with at least 7/8 on Stage A, at least 9/10 in each Stage B suite, and no stale use. Stage C would have confirmed any candidate route on 20 new tasks.

## Experiments and Results

**Table 1. Retrieval run, development cases (8 per suite; correct answers).**

| Suite | no_retrieval | always_retrieve | heuristic | model_gate | full_context | symbolic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| access | 5/8 | 6/8 | 6/8 | 6/8 | 4/8 | 8/8 |
| dispatch | 5/8 | 5/8 | 6/8 | 6/8 | 5/8 | 8/8 |

Both suites failed readiness. Access had a retrieval lift of 1/8 (2 required), and dispatch 0/8. Full-context was below the 6/8 threshold in both suites. The heuristic ceiling rule did not fire, since the heuristic was not 8/8.

**Table 2. Retrieval decisions and cost (both suites combined).**

| Policy | Searches | Missed needed retrieval (of 6) | Stale-evidence uses | Total prompt + output tokens |
| --- | ---: | ---: | ---: | ---: |
| no_retrieval | 0 | 6 | 0 | 5,494 |
| always_retrieve | 16 | 0 | 0 | 8,185 |
| heuristic | 12 | 0 | 0 | 7,631 |
| model_gate | 10 | 2 | 0 | 13,439 |
| full_context | 0 (whole corpus loaded) | 0 | 3 | 26,275 |
| symbolic | 12 | 0 | 0 | 0 (no LLM calls) |

The no-retrieval total is the sum of its two suites' prompt and output tokens (2,364 + 144 and 2,779 + 207). The model gate's total includes 6,202 tokens spent on the gate calls alone. Both missed retrievals were on expired-current cases, one per suite. The model gate used 64% more tokens than always-retrieve and 76% more than the heuristic. Full-context used 26,275 tokens and produced three stale-evidence answers.

**Table 3. Delivery follow-up (per suite, 10 development tasks, correct answers, with evidence-valid count and stale-use count).**

| Route | Access | Dispatch | Stage A nonce control (8 tasks) |
| --- | --- | --- | --- |
| chat_system | 6/10 (8/10; 2 stale) | 8/10 (9/10; 1 stale) | 8/8 |
| chat_user | 6/10 (9/10; 1 stale) | 7/10 (8/10; 2 stale) | 8/8 |
| raw_system | 7/10 (7/10; 2 stale) | 8/10 (9/10; 1 stale) | 8/8 |
| symbolic | 10/10 | 10/10 | not applicable |

No route reached 9/10 in both suites with zero stale use, so no candidate route was selected and Stage C was not run. The no-instruction negative control returned the expected `UNKNOWN` on all eight tasks. Stage B failures included abstaining on valid evidence and answering with a value that was not the target, including values from future-dated, wrong-entity, or expired-current cases.

## Discussion

The central result is negative. The model's retrieve-or-act decision did not outperform a simple observable heuristic. Relative to that heuristic, it spent fewer searches, but this saving came from skipping retrieval in exactly the expired-current cases where retrieval was needed. Its extra model calls made it substantially more expensive overall, which is the cost trade-off that a controller would have to justify. The symbolic baseline solved both suites without a language model, which means these tasks test answer-policy reliability and simple rules more than executive control.

The delivery follow-up clarifies where the failures are. The model did follow simple instructions on all three routes, so an instruction-transport failure is not supported. Evidence use, however, failed on the same kinds of cases: expired, future-dated, wrong-entity, and conflicting records. Because retrieval was held fixed in that stage, the errors came from answering with supplied evidence, not from failing to search. This matches the pilot's observation that correct routing alone does not fix answer errors.

For the architectural hypothesis, the evidence is limited. The decision model owned the retrieval choice here, but this was a prompted decision in a task that a simple resolver solves exactly. The result therefore neither supports nor refutes decision-model ownership of recall in a setting where simple rules cannot. A useful next step would isolate evidence-use judgments on minimal fresh pairs, then test a task where the observable heuristic leaves room for improvement. Any such work would need a new pre-registered protocol and fresh splits. Training a gate on the current tasks would mostly imitate the symbolic resolver, so we did not do it.

## Limitations

The task is synthetic and template-generated, with eight development cases per suite in the retrieval run and ten per suite in the delivery run. One observation per arm was collected and no repeats were run, so differences of one or two cases are not statistically meaningful. The held-out evaluation (48 cases) and the delivery confirmation stage were never executed. Only one model was tested (Qwen2.5 14B, Q4_K_M), with no training. The query was fixed to exact entity matching, so model-written queries, evidence selection, and memory writes were not tested. Evidence-valid counts measure citations rather than reasoning, and private model reasoning was not observed. Latency includes model loading and host caching. The seed and temperature were fixed, but bitwise GPU determinism is not guaranteed.

## Conclusion

On a synthetic certificate-lookup task with one local 14B model, a model-controlled retrieve-or-act gate matched a simple heuristic on accuracy, reduced searches only by skipping needed retrievals, and cost substantially more tokens. The readiness rule failed in both suites, and the follow-up found that evidence use, not instruction delivery, was the bottleneck. The fully symbolic baseline solved the task without any model call. We therefore report a negative pilot. The broader hypothesis that the decision model should own recall remains open and needs a task where simple observable rules are insufficient.

## References

- Sumers, T. R., et al. *Cognitive Architectures for Language Agents (CoALA)*, v3. https://arxiv.org/html/2309.02427v3
- Jeong, S., et al. *Adaptive-RAG*. NAACL 2024. https://aclanthology.org/2024.naacl-long.389/
- Asai, A., et al. *Self-RAG*. https://arxiv.org/abs/2310.11511
- De Sabbata, C., et al. *Rational Metareasoning for LLMs*, v3. https://arxiv.org/html/2410.05563v3
- Ollama documentation: Modelfile (system and template fields) and generate API. https://docs.ollama.com/modelfile and https://docs.ollama.com/api/generate
