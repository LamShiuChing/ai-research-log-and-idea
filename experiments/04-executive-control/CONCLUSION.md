# First executive-control decision — 2026-09-10

The retrieve-versus-act component is implemented, but this candidate experiment
**failed development readiness**. Do not start its held-out evaluation, tune on
its held-out cases, or proceed to model-written queries from these results.

The [protocol](PROTOCOL.md) was written before generation and copied into the
[run directory](results/retrieve_act_20260910T224325010154Z/REPORT.md) with hashes.
One local Qwen2.5 14B run used 96 generation calls, 61,024 actual prompt/output
tokens, and 48.4 seconds. Both 8-case development suites completed; all 48
reserved held-out tasks remain unrun. There were no API failures or invalid JSON
episodes. Seven preflight tests and all 16 post-run audit checks passed.

| Development suite | No retrieval | Always retrieve | Heuristic gate | Model gate | Full context | Symbolic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Access | 5/8 | 6/8 | 6/8 | 6/8 | 4/8 | 8/8 |
| Dispatch | 5/8 | 5/8 | 6/8 | 6/8 | 5/8 | 8/8 |

The model gate made 10 searches versus 16 for always-retrieve and 12 for the
heuristic. Its two additional avoided searches were both **missed necessary
retrievals on expired-current cases**. All three policies searched the six cases
with globally missing, stale-only, or conflicting evidence; that futility was
not generally observable before search. The model's search reduction beyond the
heuristic therefore does not show useful selectivity.

Total model-gate tokens were 13,439, versus 8,185 for always-retrieve and 7,631 for
the heuristic: approximately 64% and 76% more, respectively. The gate alone used
6,202 tokens. These totals include prompt processing and output, not just final
answers. Full-context used 26,275 tokens and had three stale-use events. The
symbolic baseline solved both suites using the same heuristic gate and mechanical
search, with no LLM calls. It has small measured CPU costs, not zero runtime cost.

The readiness rule required at least a 2/8 retrieval lift and full-context >= 6/8
in each suite. Access had only 1/8 lift and 4/8 full-context success; dispatch had
0/8 lift, with both always-retrieve and full-context at 5/8. The separate heuristic
ceiling rule did not fire because heuristic-plus-LLM was imperfect. The fully
symbolic control nevertheless establishes that these exact structured tasks do
not require an LLM. That distinction prevents misattributing answer-model errors
to a need for more elaborate retrieval control.

Observable failures show two problems to distinguish. The gate acted without
searching on both expired-current tasks. Even with the correct record supplied,
several answer calls abstained or mishandled conflicting evidence. For example,
access task `p91001/3` retrieved the valid registry code `1330`, alongside an
unofficial current value, then abstained. Thus correct routing alone cannot fix
all failures. These are observations, not explanations of private model reasoning.
Exact identical requests agreed on parsed actions; no reruns were used to choose
a favorable response.

Keep the component and its corpus as a regression fixture. The next protocol,
if pursued, should first establish reliable evidence use on fresh development
examples while retaining the symbolic baseline, then justify a task where a
simple observable resolver is insufficient. Any interface change or new task
needs a new dated protocol and fresh development/evaluation splits. Do not weaken
the heuristic, train a gate, or add query-writing on the assumption that this run
supports doing so. The broader architectural hypothesis remains open.

The prior MemRL pilot stays closed as inconclusive. Its tracked files are
unchanged relative to `e7b2f10`; this experiment uses fresh data and
does not reinterpret the historical results as supporting executive control.
