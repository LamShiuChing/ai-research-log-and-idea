# GLT — Handoff to Session 4

**Written:** 2026-08-20, end of session 3 (23:25–01:30).
**Read order:** this file → `THEORY.md` §8–§9 → `PROJECT_GOAL.md` §19–§22.
Everything stated here is measured, not planned, and reproducible from `out/`.

---

## 0. Mandate for session 4

Session 3 finished the three-specialist design and its gates. The system now answers a
free-form English question through language → logic → memory, at 1.000, and withdraws
conclusions when their premises are deleted.

The next question is what a *real* GLT needs on top of that, and the answer from §21's "what
this does not show" is not more language and not a bigger backbone. It is **the query
language**. Right now the system can express exactly one idea — follow single-valued relations
from one subject — and every remaining capability (counting, reverse lookup, comparison,
arithmetic) is blocked on extending that, not on scale.

**Update, later the same night:** Track A steps 1-4 were built and measured -- the query
language now has `follow`, `reverse`, `count`, `same` and a `none` reject, contradictions are
detected rather than silently overwritten, and PROJECT_GOAL.md §22 records the numbers. What
remains of Track A is arithmetic (G6) and rules; Track B (real English) is untouched.

---

## 1. Where the project actually is

A working three-specialist system. Every claim below is a measurement, on the same model:

| Capability | Result | Mechanism |
|---|---|---|
| Understand a question in free English | **1.000** invariance over 80 phrasings | pointer head over input positions |
| Learn a fact at inference | recall **1.000**, ~25 steps, ~1 s | one slot row, backbone frozen |
| Don't damage what it knew | 2,000 prior facts at **1.000** | disjoint token-derived addresses |
| Delete a fact exactly | **1.000** of targets | zero the row |
| Reason over what was written | **1.000** on 2-hop from free phrasing | the workspace loop |
| Withdraw what a deleted fact implied | **1.000** of derivations, **0.000** collateral | recorded premises |

The single forward pass scores **0.010** on that same 2-hop set and cannot do better in
principle. Everything above 0.010 is architecture, not training.

---

## 2. What happened in session 3, in order

### 2.1 Extraction, and two failures worth remembering

Built `glt/language.py` — a 0.89M parser that turns any wording into the canonical token line
`manager of mentor of Anna is`. That string **is** the slot address (a rolling hash over the
content tokens), which is why free text cannot address memory and why extraction is the whole
job.

**A generative head scores 0.000 on content words held out of its training data.** Zero, not
low: a softmax over the vocabulary can only emit a token whose output embedding has taken a
gradient. Fixed by pointing at input *positions* instead
([Pointer Networks](https://arxiv.org/abs/1506.03134)). Same lesson as §16's addressing
collapse — see `THEORY.md` §9.2, which now states it as a design rule.

**Only wording diversity moves held-out accuracy.** 100× parameters: nothing. 25× steps:
nothing. 4 → 32 wordings: 0.445 → 0.895. Everything fit its training data at 1.000, so nothing
was capacity-starved (`capacity.py`).

Deployed parser: invariance **1.000 / 1.000** at 1 and 2 hops over all 80 phrasings.

### 2.2 The workspace

`glt/workspace.py`, ~130 lines, no torch. A board of notes; goals and facts distinguished only
by whether the object is still `?`; §8's control rule exactly as proposed (run everything that
can fire, halt when a cycle adds nothing). Termination is structural — `infer` only posts
strictly shorter paths.

`ask.py`: **1.000 end to end** on the 3,402 §18 questions asked in arbitrary English with a
distractor name planted in the sentence.

`chat.py` rewired: questions go parser → board → slots; regex survives only for the *commands*
(`teach`, `forget`, `bits`).

### 2.3 G5a, measured for the first time

`cascade.py`. Session 2 could not ask this — Llama composed at 0.15, so only 2 of 48
consequences had ever been derived. Now all of them are.

Derivational withdrawal **1.000**, collateral **0.000**, provenance matched the ground-truth
entailment graph **3,402/3,402**. Two consequences kept the same answer word, both via a
different intermediate, so the answer-collision rate is 0.005 and continued influence is 0.

---

## 3. Design decisions

### Settled — do not relitigate

| Decision | Why |
|---|---|
| **Extraction points, never generates** | generation is 0.000 on unseen content words, structurally |
| **The parser is a separate model from the backbone** | retraining the backbone puts G2 back in play; a parser that has never seen a fact cannot pass by recalling one |
| **Control = run everything that can fire, halt on no-new-fact** | works as proposed; no learned controller, structural termination |
| **Provenance on the board, not in the slots** | makes retraction a set operation; measured 1.000/0.000 |
| **`reverse` scans; there is never an inverse index** | an index is an external fact table, and losing "facts live in network parameters" makes this project the prior art |
| **Retraction is answer-specific where a contradiction is involved** | withdrawing "guardian is Elena" must not take "guardian is Zach" with it |
| **Buy language, build exactness** | 100× parameters and 25× steps changed nothing; only training-distribution diversity did |
| Everything settled in session 2 (own the pipeline, recompute never write back, token addressing, MDL admission, no early stopping) | unchanged |

### Open — session 4 must answer

1. **Abstention is now on the critical path, not a Track C nicety.** `reverse` scans, so it
   inherits one chance to confabulate per entity read — 0.401 invented names per true edge, cut
   to 0.032 by a confidence floor and no further (PROJECT_GOAL.md §22). Every future scanning
   operation inherits the same bound, and arithmetic over counted quantities would inherit it
   twice. G4 is the fix, and it is unbuilt.
2. **Does provenance survive computed values?** If the logic specialist returns a number rather
   than an entity, the premise chain has to carry through arithmetic, or G5a stops holding for
   anything derived. `count` carries its grounds, but nothing yet composes over one.
3. **Where do rules come from?** A world-rule ("if X manages Y then Y reports to X") should be
   stored like a fact, so teaching, deleting and cascading all work on it unchanged, and the
   logic specialist stays tiny by applying whatever rules memory holds. None of it is built, and
   it is the difference between a query language and reasoning.

---

## 4. Suggested plan for session 4 — pick one track

Each step has a verify; do not start the next until it is green.

### Track A — the query language (steps 1-4 done; 5-7 remain)

| # | Task | Verify |
|---|---|---|
| ~~1-4~~ | ~~reverse, count, same, reject, contradiction~~ | done — PROJECT_GOAL.md §22 |
| 5 | **Abstention (G4).** A memory that answers a question it has no fact for is what bounds every scanning operation | `reverse` exact set match reaches 1.000 without a confidence floor |
| 6 | Arithmetic as an exact operation (G6) | **accuracy does not decay with compositional depth** — the Faith-and-Fate signature is the whole point |
| 7 | Rules stored as facts; the logic specialist applies whatever memory holds | teach a rule, use it, delete it, and watch G5a cascade through its conclusions |

### Track B — real English (the demo track)

| # | Task | Verify |
|---|---|---|
| 1 | Llama-3.1-8B 4-bit as specialist 1, constrained to emit the canonical form | invariance on wordings *nobody in this project wrote* |
| 2 | Name mapping between real words and the closed vocabulary | an unknown name is reported, not silently dropped |
| 3 | Dialogue state on the board (pronouns, follow-ups) | "what about her mentor?" resolves against the previous turn |
| 4 | Re-run `ask.py` | still 1.000 end to end |

Track B makes it *feel* like a chatbot and proves the swap is a drop-in. Track A is what makes
it a system worth having. VRAM is the practical constraint on B (ComfyUI holds ~15 GB).

### What does *not* need training

Worth stating plainly, because it is the design's main economy: the logic engine is code, the
memory needs no training to store (a write is ~25 gradient steps on one row at inference), and
the backbone is trained once. **The only component whose quality is bought with training is
language coverage** — and §19 measured that parameters and compute do not buy it, so the answer
there is a pretrained model rather than a bigger local run.

---

## 5. Run it

```bash
python chat.py                 # the demo: ask anything, teach, forget, board, where, bits
python ask.py                  # 0.010 -> 1.000, from free-form English (3,402 questions)
python cascade.py              # G5a: 1.000 withdrawal, 0.000 collateral
python extract.py              # the parser: heads, wording sweep, invariance gate (~7 min)
python capacity.py             # parameters and steps buy nothing (~35 min)
python operations.py           # reverse / count / same against ground truth
python chain.py                # session 2's composition result, decomposition handed in
python -m pytest tests/ -q     # 183 tests
python -m pyflakes glt/*.py tests/*.py *.py
```

| Path | What |
|---|---|
| `glt/language.py` | specialist 1 — wording → canonical line; `head` = point \| generate |
| `glt/workspace.py` | the board, the control loop, provenance, retraction (**no torch**) |
| `glt/memory.py` | `MemorySlots`; `read` = token (the others are dead ends, kept with their numbers) |
| `glt/model.py` | nanoGPT fork (bit-identical) + slots + `address()` + `hidden()` |
| `glt/admission.py` | MDL surprise gate, in bits |
| `chat.py` | the demo, rewired through the workspace |
| `ask.py`, `cascade.py`, `extract.py`, `capacity.py`, `operations.py` | session 3's five measurements |
| `out/online.pt` | **the model to use** — token-addressed, 16384 slots, 200 facts written |
| `out/parser.pt` | **the parser to use** — pointer head, 40 wordings, all words |
| `out/memory.pt` | whatever you taught it interactively; delete to reset |

---

## 6. Gotchas that cost time this session

- **A strided slice cannot be `.view()`ed.** nanoGPT's loss reshapes with `.view`, so a batch
  built by slicing a padded tensor must be `.contiguous()` first.
- **Left-padding breaks a model with learned absolute positions.** Generative decoding is
  batched by question length instead; the pointer head pads right and masks, which is safe
  because the query is gathered at the true `SEP` index.
- **Two meanings of `?`.** It is the open-goal marker *and* a vocabulary token, so a memory that
  answers `?` reads as an abstention. That is the behaviour you want, but it was an accident
  until it was tested.
- **A demo that checks itself against ground truth must update that ground truth when you
  teach it.** `chat.py` called itself wrong for correctly using a fact the user had just taught.
- Session 2's gotchas still apply: non-ASCII needs `encoding="utf-8"`, ComfyUI holds ~15 GB of
  VRAM, verify that an edit landed, identical numbers across different settings mean a bug.

---

## 7. Dead ends — measured, do not repeat

| Tried | Result |
|---|---|
| Generative decoding of the canonical form | 0.000 on any content word held out of training — structural, not a data shortage |
| More parameters for the parser (0.14M → 14.88M) | no change on held-out wordings; everything already fits at 1.000 |
| More steps for the parser (4k → 100k) | no change; the parse rule does not grok the way composition did |
| Filtering a scan by slot ablation ("is anything really stored here?") | keeps 24 of 34 invented edges — the address of a non-fact still lands on a slot carrying something. The confidence floor works; this does not |
| Everything in session 2's table (softmax/gated/threshold reads, hidden-state addressing, writes confined to fresh rows, slots on a fully-trainable model) | unchanged |
