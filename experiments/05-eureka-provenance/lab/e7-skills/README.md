# E7 — continual SKILL learning

E5 settled facts: the store learns them with zero forgetting. But a new
*ability* — here, 2-digit addition — cannot live in a datastore; it must
enter the weights, which E5 showed smears old abilities. E7 measures that and
tests the architectural cure.

Skill A: fact QA (200 entities). Skill B: addition, trained on 3,000 pairs,
**evaluated on held-out pairs** — so B accuracy measures the skill, not
memorization. Four arms:

| arm | phase 2 | prediction |
| --- | --- | --- |
| base | — | A high, B ~0 |
| naive | train all weights on B | B learned, A forgotten |
| replay | B + 25% A | both, at retraining cost |
| **adapter** | freeze base entirely; train zero-init bottleneck adapters (fork: `adapter_dim`) | B learned by the adapters, A structurally protected |

The adapter arm is the architecture-level answer: new skills go into small
added modules; the base is read-only. (Cost: adapters stay active at
inference; a per-skill routing decision becomes the next question if arms
multiply.)

## Run

```powershell
python prepare.py
cd ..\gpt
python train.py ..\e2-abstention\train_facts.py --compile=False --dataset=../nanogpt/data/e7_a --out_dir=out-e7-a
# copy out-e7-a to out-e7-naive / out-e7-replay, then resume-train each; adapter arm:
python ..\e7-skills\adapter_train.py out-e7-a out-e7-adapter
python ..\e7-skills\eval.py out-e7-a out-e7-naive out-e7-replay out-e7-adapter
```

## Results

2026-08-31, two design iterations.

**v1 (base never saw digits): adapter arm 0/0.** The frozen base's digit
embeddings were random — never trained, since skill A's corpus had no numbers
— and adapters can't fix frozen embeddings. Lesson: **the capable base
requirement reaches down to token embeddings**; a base must at least
represent the primitives of future skills. (Real pretrained LLMs get this
from breadth; fixed here with benign counting lines in the base corpus.)

**v2 (digit-competent base):**

| arm | skill A (facts) | skill B (heldout addition) |
| --- | --- | --- |
| base | 100 | 0 |
| naive | **0** | 100 |
| replay | 100 | 100 |
| adapter (always-on) | **0** | 63 |
| **adapter + routing** | **100** | **63** |

- Naive is total skill destruction (worse than facts' 15% in E5).
- Adapter alone learns real generalizing arithmetic through a frozen base —
  but destroys A anyway: it fires on every input and nothing in a B-only loss
  enforces silence on A. **Frozen weights protect parameters, not behavior.**
- Routing (engage the adapter only on digit-bearing inputs) restores A
  completely at zero training cost. Append-only skill learning: base
  read-only, no old data kept, each skill in its own module, a router
  deciding engagement.

Open: B's 63 vs replay's 100 (adapter capacity/steps ceiling through a frozen
base); learned routing instead of a hand rule; multi-skill composition (E7b).

## E7b — third skill + learned routing

Subtraction as a second adapter (`e7_c`, held-out pairs again), and the hand
rule replaced by a **learned router**: a linear probe on the frozen base's
last hidden state, trained on 450 task-labeled question prompts (inputs only —
no answers, no old training data), 300 optimizer steps.

| | result |
| --- | --- |
| router accuracy (fresh questions) | **150/150** |
| routed: colors (base) | **100/100** |
| routed: heldout addition (adapter B) | 63/100 |
| routed: heldout subtraction (adapter C) | **97/100** |

Three skills, one frozen base, two ~590K-param modules, and a 1,155-param
router — nothing forgotten, nothing retrained, routing learned not written.
The frozen core's representations separate task types so cleanly that a
linear readout suffices — the "capable base" earning its keep again.

Notes: subtraction's 97 vs addition's 63 likely reflects task difficulty
(sums carry into 3 digits; differences stay short), not adapter order. Router
training needs example *questions* from every task (labels = task id only) —
a far lighter footprint than replay's answers-and-all, but not zero; a
router that detects "none of my skills" is the next open edge.
