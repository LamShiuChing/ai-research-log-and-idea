# unified/ — the Mind: every validated organ, one system

`mind.py` wires the day's results into a single class; `serve.py` + `index.html`
put a chat and a teach box on it (port 8342).

| organ | from | role |
| --- | --- | --- |
| consolidated core | E9 | colors + arithmetic living in frozen weights |
| editable memory | E6/E5 | kNN store over the fact base; `teach()` adds facts live, zero training |
| tool layer | E8 | the oracle audits arithmetic answers at inference; corrects the core when they disagree |
| self-awareness | E8 | self-calibrated novelty detection; unknown skills are declined honestly and logged to `backlog.jsonl` for future adapter training |

Smoke test (2026-08-31):

```
What is the color of Bardol Heltor?  -> silver   [core + memory]
What is 47 times 8?                  -> 376      [core (tool-verified)]
What is the capital of France?       -> declined [novelty -> backlog]
teach: Glorbo Zenmax = crimson       (API call, no training)
What is the color of Glorbo Zenmax?  -> crimson  [core + memory]
```

The provenance channel (EP line) lives in the sibling EPX model (different
vocab/domain); the fork supports all organs in one config — combining them in
a single trained base is the scale-up run's job.

Calibration note: novelty references must cover the *variance* of each known
skill (color questions scatter per entity name — 120 references, not 10), and
the threshold is set from fresh in-distribution questions. Startup builds the
store and calibrates (~1 min).

## Run

```powershell
cd lab\gpt
python ..\..\unified\serve.py    # http://localhost:8342
```

## Mind v2 (SmolLM3-3B, `serve_v2.py`, port 8344) — graded memory

`store.py` holds every memory claim as `question -> {source: {a, grade}}`:

| grade | source | resolution |
| --- | --- | --- |
| G0 verified | oracle | outranks all |
| G3u you told me | `user` (regex / `/api/remember` without `source`) | outranks sources and guesses |
| G1 external | any named `source` | E12b: trust-weighted vote; trust = Laplace accuracy vs anchored (G0/G3u) claims |
| G2 unconfirmed | `chat` (LLM extraction) | lowest; never overrides; shown as a guess |

The relevant slice enters the system prompt with its label; `GET /api/memory` shows the
resolved facts. `test_store.py` covers migration, override rules, trust, the vote, forget.

Measured 2026-08-31 (isolated copy, 9 probes, one sampled draw each):
- a chat-extraction guess (`What is my name = Mochi`, emitted on a translation turn) no longer
  overwrites the user-taught name — the exact corruption seen before this change;
- contested `hometown`: wiki (trust 0.80) vs two lying sources → answered Hong Kong, trust cited;

Extraction (`extract.py`), measured 2026-08-31 on 27 probes after the fix: whole-message
templates only (`my name is not important` no longer saves `not`; `which colour do I like
best?` is answered, not eaten); the core's extractor runs only on first-person statements or
<=3-word replies, and a value must be <=3 of the user's own words — requests (translate, write
code, "tell me about basketball") no longer produce facts. A first-person statement is saved as
`you told me`; a short context reply (`basketball` after "what's my sport?") as a guess.
`you told me` facts are stated without caveats. Each `/api/ask` reply lists the memories it
consulted (`facts`). Still open: a bare "yes" does not promote a guess; the 3B core is verbose;
retrieval past 6 memories is untested.

## Mind v3 (`mind_v3.py`, `serve_v3.py`, port 8346) — model-driven tool loop: measured, negative at 3B

Everything rule-based removed: no routing regexes, no extraction templates, no command
syntax. The core sees four tools through its own chat template (`xml_tools`: recall, calc,
remember, forget) and drives a call loop; the store still grades by evidence (a remembered
value must appear in the user's words) and an answer is "verified" only if every number in
it came from calc. `test_tools.py` covers the calculator sandbox, call parsing, number check.

Three runs of the same 38 probes, 2026-08-31, three distinct failure modes:

1. Greedy, facts injected in the system prompt: the core wrote the call JSON *without*
   `<tool_call>` tags → 0 tool calls executed; every math answer was its own guess
   (12*12+47*8 → 1016, 23*17-95 → 322, correctly labeled "unverified numbers"). Once one
   tagless answer entered history, every later turn imitated it.
2. Facts block moved last, bare-JSON calls accepted, evidence gate on remember: the core
   called `remember` to answer "what's my name?", was refused, and lectured the user about
   the function for six turns (37-75 s each); afterwards it stopped calling tools entirely
   and invented sources ("Melbourne [source: Wikipedia]", "2023 [source: Unix timestamp]").
3. `recall` tool instead of injection, repeat-call guard, model-card sampling (0.6/0.95):
   the core parroted the instruction text ("I don't know yet and ask to be told.") for 17
   turns and emitted `Calc: {...}` / `Recall: {...}` — no protocol, 0 calls.

In isolation (one turn, short prompt) the same core emits correct tagged calls for
`what is 47 times 8` (calc) and `my sister is called Anna` (remember) — the failure is the
multi-turn loop, not the format. Conclusion: at 3B, model-driven control is worse than v2's
rules on every category (v2: memory 11/11, single-op math verified, chains 1/4). The harness
is kept for the step that could change this — training the core on its own oracle-verified
tool traces (E8's loop applied to tool use) — or a larger core.
