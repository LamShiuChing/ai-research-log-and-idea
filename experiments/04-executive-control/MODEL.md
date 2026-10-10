# The executive-control model we are building

Status — 2026-09-10: policy specification and experimental scaffold. **No learned
executive-control weights exist yet.** Qwen2.5 14B is the current frozen base
model used to test the interface. It is not the name of a newly trained controller.

## What the model owns

The executive model is an action policy:

`policy(parameters, goal, current evidence, prior tool results, remaining budget) -> action`

Its output should select an external answer/abstention or an internal information
operation. The first controlled action set is retrieve versus act, with the query
and retrieval limit held fixed. Later action sets can add model-written queries,
individual reads, and evidence acceptance/rejection. Those changes require
separate tests so an apparent gain can be attributed to a specific decision.

Memory stores immutable records and performs mechanical search. It never predicts
whether recall is useful, learns outcome utility, or chooses which evidence the
agent should believe. The host enforces budgets and validates action syntax. The
evaluator alone sees target answers and success labels.

The policy must distinguish three questions that a single "insufficient evidence"
flag cannot resolve:

1. Does the evidence already justify an action?
2. Could a search supply information that changes the best action?
3. Is the expected improvement worth the full cost of that search and subsequent
   model calls?

Missing evidence can warrant search, while a known unresolved contradiction may
warrant abstention. A fact existing somewhere in storage is not information the
policy may use before searching. Training and evaluation must respect that boundary.

## What learning would optimize

For an observation `o`, compare the expected task utility after acting now with
the expected utility after the fixed retrieval operation. Subtract the operation,
routing, and answer-generation costs. A supervised first gate can predict this
difference; a threshold selects retrieval. This is a proposed learning target,
not a claim that the frozen Qwen gate currently estimates it.

Training labels must come from paired act/retrieve outcomes on training-only
tasks under the same executor and budget. They must include failed searches,
unusable results, and answer-model failures. Repeated training-only rollouts may
be needed when outcomes vary. Case categories, correct codes, or labels derived
from held-out successes cannot become model inputs. Report label-generation and
training cost alongside inference cost. Do not label every missing-current case
"retrieve" merely because a hidden generator says it requires memory.

Candidate implementations are a small supervised decision head over a frozen
observation encoder or an adapter on the decision model. Architecture choice is
deferred until a task supports a useful learning signal; these files contain no
trained head or adapter. The model should eventually be able to act directly in
the same policy call when possible, because a compulsory separate gate call can
consume more than the searches it saves. That change is distinct from the first
binary-gate experiment and needs its own comparison.

## What exists and what blocks training

- `experiment.py`: passive storage, observable heuristic and symbolic baselines,
  a prompted binary policy, budgeted execution, scoring, and complete traces.
- `instruction_delivery.py`: a bounded check of instruction delivery and the
  answer model's ability to use supplied evidence, with fresh splits and controls.
- v1: no demonstrated gate advantage; the model gate missed necessary searches
  and used more tokens than the heuristic. The certificate task is symbolic.
- v2: instruction controls passed, but evidence use remained unreliable across
  delivery paths. No route qualified for fresh confirmation.

Before training, isolate the failing evidence judgments on fresh minimal pairs
and establish a reliable fixed executor. Then require a nontrivial environment
where the strong observable heuristic and symbolic baseline leave a meaningful
quality/cost opportunity. Training a gate on the current symbolic certificates
would mainly imitate cheap code or fit executor failures, rather than establish
useful executive control. A successful later trained controller must be tested
on fresh tasks and compared with all simple baselines under complete cost accounting.

The original research hypothesis remains open: decision-model ownership of recall
is an architectural boundary to test, not a result proved by these diagnostics.
