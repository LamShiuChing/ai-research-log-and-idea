# Instruction delivery and evidence use — protocol v2, 2026-09-10

Written before v2 inference. This is a diagnostic follow-up to the failed
retrieve-versus-act readiness check, not a retry of its held-out evaluation.
Freeze this document, sources, complete requests, and dataset hashes before calls.
All existing result directories and MemRL artifacts remain unchanged.

## Question

Do rule-bearing system instructions reach the existing Qwen2.5 14B model in a
usable form through the current local chat/template path? Are evidence-use
failures reduced by delivering the identical instructions via another path?
The previous traces alone do not establish an instruction transport defect.

The local `/api/show` response exposes a default system message and a template
that renders `.System` before iterating through `.Messages`; its message loop
has user/assistant/tool branches. This motivates a diagnostic, not a conclusion
about how the server populates those variables. Save the complete template and
model metadata. Ollama documents [system/template fields](https://docs.ollama.com/modelfile)
and [raw generation](https://docs.ollama.com/api/generate). Raw ChatML below uses
the exact role delimiters observed in this model's template. No model template,
service setting, or weights are modified.

## Stage A: simple instruction delivery controls

Eight independent nonce tasks, seed 101001. Each instruction specifies a newly
generated random marker and requests `{"answer": "<marker>"}`. The user payload
asks for the marker, or `UNKNOWN` if none was specified. The output schema permits
any string; it never exposes the marker. Four arms, randomized within each task:

1. `chat_system`: `/api/chat`, instructions in the system message, payload in user.
2. `chat_user`: `/api/chat`, no supplied system message, same instructions followed
   by the same payload in one user message. The model's default system may remain.
3. `raw_system`: `/api/generate`, `raw=true`, explicitly serialized
   `<|im_start|>system\n{instructions}<|im_end|>\n<|im_start|>user\n{payload}<|im_end|>\n<|im_start|>assistant\n`.
4. `chat_none`: `/api/chat`, payload alone. Expected output is `UNKNOWN`.

Interpretation: positive arms test instruction-conditioned behavior; the negative
arm tests guessing or unintended marker leakage. Differences are evidence about
delivery paths on this host, not proof of an internal server mechanism. These
are eight distinct marker tasks, not repeated samples of one prompt.

Stop all inference after Stage A if negative control < 7/8 or all positive arms
< 7/8, or if infrastructure failures exceed the run limit. Otherwise proceed.

## Stage B: evidence-use development

Twenty fresh certificate tasks: 10 access (one field), 10 dispatch (two fields).
Seed 101101. One of each case per suite: current sufficient, memory required,
expired current, unofficial current, future current, wrong-entity current,
missing everywhere, stale-only memory, conflicting valid memory, current with
stale memory. Codes use a fresh `C` plus six-hex-digit alphabet rather than v1's
numeric codes; entity and record prefixes are distinct. This avoids reusing any
v1 identities/answers without reading its held-out examples. Dates and symbolic
semantics remain the same. Causal comparisons are within v2, not versus v1 scores.

All three positive delivery arms run, including any weak Stage A arm, with the
exact original v1 answer instruction, schema, and separated MEMORY/CURRENT TASK
payload. This holds evidence, order, semantic instructions, and answer interface
constant across the delivery comparison. Fixed mechanical search returns up to
8 same-entity records in record-ID order, including unofficial/stale distractors.
The host never filters evidence. There is no retrieval decision in this diagnostic:
retrieval is always provided to isolate the answer-policy prerequisite. A symbolic
resolver using the same supplied records is included at zero LLM calls.

Keep only routes with Stage A >= 7/8, development >= 9/10 exact answers in EACH
suite, and zero stale-use events as candidates. If none qualify, stop. A symbolic
failure stops the study as a task/harness problem. No prompt edits after outcomes.

## Stage C: fresh confirmation, conditional

Twenty further tasks, seed 101201, with disjoint entities/record IDs/codes and the
same scenario balance. Run ALL three positive routes, regardless of candidate
selection, when at least one Stage B candidate exists. This avoids favorable-arm
only reporting. A route is supported for a future controller protocol only if it
was a Stage B candidate and achieves >= 9/10 in EACH confirmation suite with zero
stale use. This is interface confirmation within a synthetic family, not transfer.

If more than one route qualifies, prefer `chat_system` when it qualifies; otherwise
`raw_system`, then `chat_user`. Prefer maintaining a distinct system role rather
than conflating system policy with user input. This order is fixed before calls.
No certificate gate, query-writing, or training run follows automatically. Save
the evidence and a concrete follow-up decision first.

## Inference, budget, logs and scoring

Only existing loopback Ollama, `qwen2.5:14b`, digest
`7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6`.
Temperature 0, seed 42, context 16,384, output cap 96 for all arms. No downloads,
training, embedding calls, cloud inference, or permanent runtime changes.
Python standard library, using v1's passive store, instruction and scoring code
read-only. Raw outputs go through the same strict parser as chat outputs.

At most 152 generation calls (32 control + 60 development + 60 confirmation),
15 minutes, 2,490,368 prompt/output tokens. At most 1 generation and 1 search per
episode, 16,384 total tokens and 60 seconds. Pre-dispatch UTF-8 payload plus 512
template allowance and output reserve must fit context. All failures count as
incorrect; no retries or fallbacks. Stop on 3 infrastructure/budget failures.
Invalid JSON counts as a failed episode without a free replacement call. Report
partial runs. Always write attempted requests, raw responses, timing and errors.

Before inference, validate nonce isolation, split separation, search coverage,
reference answers, exact instruction/payload equivalence across routes, raw
serialization, and context bounds. Source/model/protocol hashes accompany the
run. Record status and request start timestamps so freeze-before-run is auditable.

Report exact success, supporting-evidence validity, stale use, failures, total and
cached prompt tokens (cached counts are a subset, not added again), generated
tokens, latency including load, search time, loaded bytes/IDs, and model calls.
Report paired semantic outcomes for the same tasks. Stage C paired win/loss/tie
counts and exact two-sided discordant-pair binomial p-values are descriptive;
multiple comparisons and small, shared-template samples preclude broad claims.
Inspect specific observable failures; do not infer private reasoning.

## Relation to the executive controller

The controller will own retrieve/act decisions and evidence judgments. Passive
memory does not acquire a policy in this diagnostic. A validated instruction
transport is a prerequisite for testing or training the decision model: otherwise
we risk attributing an interface failure to its ability to exercise control.
Even perfect performance here leaves the certificate task symbolically solved;
do not present an interface repair as a learned executive-control model.
