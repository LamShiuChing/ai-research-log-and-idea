# Retrieve-versus-act protocol v1 — 2026-09-10

Written before any executive-control model calls. Base: `e7b2f10`.
Status at creation: frozen candidate-task experiment, including development gates.
The run copies this document, executable sources, prompts, config, model metadata,
and dataset hashes before its first generation. Results never overwrite a run.

## Question and scope

Can frozen Qwen2.5 14B decide whether to retrieve, preserve exact task success,
and reduce total computation compared with strong simple policies? This is a
prompted binary controller, not a trained gate or a test of the entire design.
The decision model chooses `act` or `retrieve`. On retrieval the host executes
the same fixed entity query, then the same model selects an answer or abstains.
Memory only stores immutable records and performs deterministic exact-key search.
It does not see evaluator labels, decide relevance, filter by authority/validity,
or learn utility. Model-authored queries and writes are deferred.

## Tasks and splits

Two separately generated suites: a one-field access certificate and a two-field
dispatch certificate. These are distinct coverage checks within one synthetic
task family, not independent evidence of broad domain transfer. Each task has a
fresh random entity, random four-digit opaque codes, requested fields, an as-of
date, and current records. Records have IDs, source, issue/expiry dates, field,
entity, and value. Registry certificates are final for their validity interval;
valid current certificates cannot be silently superseded in storage. Conflicting
valid registry values require abstention. Expired, future, and unofficial records
are unusable. Answer all requested fields in their given order, joined by `/`,
or `ABSTAIN` if any field lacks unique supported evidence.

Eight evaluator-only scenarios per suite: current sufficient; memory required
(dispatch supplies one of two fields currently); expired current replaced by
valid memory; unofficial current corrected by memory; missing everywhere; stale
memory only; unresolved valid conflict; current sufficient with stale distractors.
The model sees no scenario, split, expected answer, need label, or task ID.
All retrieved records are exposed without oracle filtering. Shared query is exact
entity equality, limit 8, stable record-ID order; all matching records fit this
limit. Full-context reads the suite's entire fixed corpus, including distractors.

Development: 8 cases per suite, seeds 91001 and 91002. Conditional evaluation:
24 cases per surviving suite (3 of each scenario), seeds 92001 and 92002.
Entities, record IDs, and codes are disjoint across splits/suites; all are newly
generated independently of the MemRL data. Corpora are fixed and common to arms
within each suite/split. There is no cross-case conversation or learning.
All datasets are saved before calls, but held-out outcomes are neither run nor
used to revise decisions. Frozen seeds are reproducibility, not an access-control
claim: the implementing researcher can inspect generated data. No prompt tuning
or repeat-until-success is allowed in this version.

## Arms

1. `no_retrieval`: current evidence, one answer call.
2. `always_retrieve`: fixed search, one answer call.
3. `heuristic`: retrieve unless every requested field already has exactly one
   distinct currently valid registry value. This checks entity, source, dates,
   completeness, and conflicts using only current observations; one answer call.
4. `model_gate`: same Qwen model chooses act/retrieve in one JSON call; then one
   answer call with current evidence and any fixed search result. No hidden
   reasoning transcript is requested or used.
5. `full_context`: every record in the shared suite corpus, one answer call.
6. `symbolic`: the same observable heuristic gate and search, followed by a
   deterministic validity/uniqueness resolver using only supplied evidence.
   This is a deployable domain-specific baseline, not an evaluator-label oracle.

All LLM arms share the exact answer system prompt and JSON schema. Symbolic uses
the same answer semantics and budget ceilings with zero LLM calls. Randomized
arm order within each task and randomized task order reduce order effects;
model seed 42 and temperature 0. One observation per task/arm, no pseudo-replicates.

## Budgets and failures

Use only the existing loopback Ollama endpoint `http://127.0.0.1:11434` and model
`qwen2.5:14b`, digest
`7cdf5a0187d5c58cc5d369b255592f7841d1c4696d45a8c8a9489440385b22f6`.
Python 3.10 standard library; no package, model, service install, cloud inference,
embeddings, training, or runtime changes. Local exact search avoids an embedding
model confound. Capture Ollama version and model metadata before calls.

Same hard per-episode ceiling for every arm: 2 generation calls, 1 search,
32,768 cumulative prompt-plus-output tokens, and 120 seconds. Each call reserves
up to 16,384 context tokens with 96 maximum output tokens. Conservatively require
UTF-8 message byte count + 256 wrapper allowance + 96 <= 16,384 before dispatch;
check actual returned counts and reject length truncation. This leaves room for
both calls within the common total ceiling. Generation timeout is the lesser of
60 seconds and remaining episode/run time; no retries or fallback answers.

Whole run: at most 384 generation calls (96 development + 288 conditional
evaluation), 30 minutes, 12,582,912 counted prompt/output tokens. Stop on 3 API or
budget failures; record all attempts, partial outcomes, and stopping reason.
Invalid JSON/invalid actions count as failed episodes; a bad gate does not trigger
a free heuristic fallback. First-call loading is included; latency is descriptive
and host/cache-dependent. CPU search latency and returned payload bytes are logged.

## Development gates and stopping rule

Run all arms on the 16 development cases, then decide separately for each suite:

* Require symbolic 8/8 (otherwise harness/task failure), at least 95% valid model
  episodes, and both always-retrieve and full-context >= 6/8.
* Require always-retrieve to beat no-retrieval by >= 2/8, verifying retrieval has
  a task effect. Require at least two sufficient-current and two memory-needed
  cases, with mechanical search covering their requisite records.
* Retire a suite as cheaply resolved if heuristic is 8/8 with >= 25% fewer searches
  than always-retrieve. Do not weaken the heuristic. This retirement takes
  precedence over starting evaluation, even if the model gate also looks good.
* Otherwise run that suite's 24 frozen held-out cases once if readiness passes.
  If neither suite survives, finish with a development report and unrun held-out
  manifest. Do not silently substitute another task or tune this one.

## Metrics and useful-effect threshold

Report exact success (including required abstention), correct evidence coverage,
invalid/API failures, actual prompt/output tokens including gate, calls, wall
latency, retrieval frequency, gate latency/tokens, search latency and payload.
Evaluator-only `needs_memory` means current cannot answer a non-abstention target
but search supplies it. Report missed retrieval over that denominator. Unnecessary
retrieval means current evidence already uniquely suffices; separately report
searches on globally missing/stale/conflict cases, whose futility is generally
unobservable before search. Stale use is citation of an expired/future record or
emitting its value when that value is not the target; report unofficial evidence
separately through evidence validity. Never equate no search with no memory cost
for full-context: report its loaded records/bytes and tokens explicitly.

Held-out useful effect requires, in each surviving suite: model-gate success no
more than 1/24 below always-retrieve AND heuristic, >= 25% fewer searches than
always-retrieve, >= 10% fewer total tokens than always-retrieve, and no increase
in stale use. Report comparisons with full-context and symbolic even if they
dominate. Beating only always-retrieve is not a demonstrated controller advantage
if a cheaper simple baseline matches/exceeds quality. At this sample size these
are screening thresholds, not proof of noninferiority. Report paired win/loss
counts, and descriptive paired bootstrap 95% intervals within suite (10,000
resamples, seed 93001); shared templates/corpus limit independence. Development
scores get counts only, with no confirmatory claims. No query-writing follow-up
is run unless a separately written protocol is justified by informative evidence.

All original MemRL files/results remain unchanged. See `PRIOR_WORK.md` for overlap
with CoALA, Adaptive-RAG, Self-RAG, and Rational Metareasoning.
