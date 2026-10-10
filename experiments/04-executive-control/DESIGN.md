# Decision-model-owned retrieval

Status: broader design draft; the first binary controller is now implemented.
The [first locked protocol](PROTOCOL.md) was executed through development and
[stopped at its readiness gate](CONCLUSION.md); no held-out evaluation ran.
An [instruction-delivery diagnostic](results/instruction_delivery_20260910T225955352340Z/REPORT.md)
also stopped after development. See [MODEL.md](MODEL.md) for the intended learned
executive policy and the distinction from the current prompted scaffold.
Selected on 2026-09-10 after closing the MemRL pilot. This follows the
executive-controller branch in roadmap v1.5, Section 12.

## Researcher's hypothesis

The memory component stores things; the decision model supplies the active
process that connects a current goal to remembered experience. Retrieval intent
and the judgment of what to use should belong to the decision model, rather
than to a separate memory policy optimizing its own utility scores.

Here, "internal" means owned by the decision model's action policy. The model
may invoke an external database tool. It does not mean storing every record in
model weights, or requiring a new neural architecture. The "living mind" analogy
motivates the design; biological or consciousness claims are outside this test.

## Component boundary

The store preserves records, identifiers, source, and timestamps. It supports
deterministic read/search operations and returns candidates. It does not decide
when an agent should remember, rewrite its goals, or learn outcome-based utility
scores. An index can rank query matches mechanically without owning task policy.

The decision model observes the task and chooses an action: answer, search with
a query, read a candidate, or abstain when evidence is insufficient. After a
read it can accept or reject that evidence and choose another bounded action.
The host enforces the action budget and logs calls; it does not choose relevant
memories for the model. The evaluator scores the final action separately.

Memory writes and consolidation are deferred. A fixed common corpus isolates
retrieval control before introducing a second decision problem.

## First bounded question

Can the same local model choose when to consult memory while preserving task
success and reducing unnecessary retrieval, compared with fixed policies?

Develop a small inspectable text task with cases where current evidence is
sufficient, cases requiring a stored fact, and cases with missing or outdated
evidence. Records must contain observable provenance and timestamps when those
are needed to resolve a conflict. Do not give the controller hidden relevance
labels, evaluator answers, or a field revealing the case category.

Compare no retrieval, always retrieve using a fixed query, a simple observable
heuristic gate, and model-controlled retrieval using the same corpus and search
tool. Include full-context access when the corpus fits, and a symbolic control
when the task admits one. These are strong baselines, not presumed inferior ones.

First isolate retrieve-versus-act with a shared query and retrieval limit. Only
if that decision is informative, separately test model-written queries and
candidate selection. Changing all of them together would obscure the cause of
a result. The initial binary gate tests only part of the broader hypothesis.

Use the same model, available evidence, answer tools, and hard total budget.
Count controller calls and search costs as well as answer generation. Report
success, actual tokens, latency, retrieval frequency, failures to retrieve when
needed, and use of stale evidence. Compare cost and quality together.

## Before execution

1. Review the roadmap's CoALA, Adaptive-RAG, and Rational Metareasoning sources
   against the precise question; establish overlap before claiming novelty.
2. Implement and validate the environment, passive store, and fixed baselines.
3. Use development cases to check that retrieval sometimes changes the outcome
   and that the model understands the interface. Retire a trivial task if a
   cheap heuristic completely resolves it.
4. Write a dated run protocol with independent suites, fresh splits, model
   digest, prompts, compute ceiling, useful-effect threshold, and stopping rule.
5. Freeze choices before evaluation. Preserve all outcomes and log the observable
   action sequence; private internal reasoning is not required evidence.

The original design checkpoint included no model runs. The subsequent bounded
implementation used the existing local setup under its separate protocol, with
no training, service installation, or cloud inference. See the
[component README](README.md) for reproduction and the preserved first outcome.
