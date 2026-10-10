# EP5 — does learned source trust transfer to a pretrained 3B core?

Status: pre-registered 2026-08-31; code in `prepare.py` / `train.py` / `eval.py`.
Step 0 (20-step dry run, A2): **10.1 GB peak, 0.20 s/step** — a full arm is 1,310 steps,
~4.5 min. A3 is implemented as the same zero-init source table applied to the first token
only (a prefix tag), not via PEFT's `trainable_token_indices` — equivalent, one code path.

EP/EP2/EPX showed a from-scratch 10–89M model with a per-token source embedding
learns which sources to trust from a small QA anchor, with no reliability labels,
and generalizes that trust to entities it had no QA for. Everything the project
calls its differentiator rests on that. It has never been tested on a pretrained
model, and it has never been compared against the obvious baseline: an instruct
model that is simply *shown* the sources in its prompt.

## Question

On SmolLM3-3B (hidden 2048, 36 layers, tied embeddings, bf16 — from the cached
`config.json`), does a small trained provenance channel produce trust-weighted
recall on held-out entities, (a) better than the same fine-tune without
provenance, and (b) better than no training at all with sources in context?

(b) is the result that can change the project's direction: if the prompt-only
baseline matches the channel, provenance-in-weights is not a differentiator at
this scale.

## World (synthetic — real facts are contaminated by pretraining)

- 500 invented countries (pronounceable nonsense names), each with an invented
  capital (unique nonsense city) and a region from a fixed 6-way set.
- Two sources, EP2/EP3 structure: S1 truthful on capitals, lies on regions; S2
  the reverse. Balanced, so only (source, topic) trust answers both topics.
  3 template reps per (entity, topic, source) → 6,000 claim lines.
- Anchor: QA lines for 400 entities (both topics, 3 reps → 2,400 lines).
  100 entities held out: claims only, never QA.
- Lies are drawn from the pool of other entities' values (plausible, as EP3).
- Data seed fixed; the same generated files feed every arm.

## Arms (same data, same seed, same eval)

| arm | training | provenance | tests |
| --- | --- | --- | --- |
| A0 prompt-only | none | source names in the prompt, anchor QA in context, then the held-out question | what the instruct core already does in-context |
| A1 LoRA, no provenance | LoRA on claims + anchor QA, source stripped | none | EP's control |
| A2 LoRA + additive channel | same LoRA | `src_emb: nn.Embedding(3, 2048)`, zero-init, added to every claim token's embedding; source 0 (zero row) on QA and at eval | faithful port of EP's `wse` |
| A3 LoRA + prefix tag | same LoRA | the same source table added to the first token of each claim only | the cheap variant a practitioner would try |

LoRA: rank 16 on q/k/v/o projections, all 36 layers. Zero-init means every
trained arm starts bit-identical to the base.

## Eval (forward-only scoring, no generation parsing)

For each held-out entity × topic: score the three candidates — truth, the
source's lie, a random other value — by summed log-prob of the answer tokens
after `Q: What is the capital of X? A:`; argmax. Report truth / lie / other
counts per topic, exactly EP's table. Also report anchored train-QA recall
(the model must at least hold what it was shown) and a **regression check**:
accuracy on 40 real countries' capitals (the EP3 list), scored the same way,
before and after — the LoRA must not damage what the core knew.

## Pre-registered criteria

- A1 heldout truth ≈ 50% or mostly "other": no provenance, no way to resolve
  the balanced conflict.
- **Transfer is real** if A2 − A1 ≥ 20 points on heldout truth for *both*
  topics with lie rate ≤ 10%, and the regression check drops ≤ 2 points.
- **Not a differentiator** if A0 ≥ A2 on heldout truth: the in-context core
  already weighs sources; write that down as the finding.
- A3 vs A2 is reported, not gated: whether a per-token channel beats a prefix tag.
- One seed first for all four arms. If A2 − A1 clears 20, rerun A1 and A2 on
  two more data seeds before any claim (the lab's single-seed habit stops here).

## Budget and risks

- Measured (step 0 and the pass): 8,400 lines × 5 epochs at batch 32 = 1,310 steps per
  arm, 0.15–0.20 s/step, 10.1 GB peak; ~4 min per trained arm. The pre-run estimates
  (25 min/arm, ~10 GB) were off on time, right on memory.
- LoRA stores knowledge poorly; anchored train-QA recall is the canary (it passed: 92–97%).
- The source table lives outside PEFT (a plain trainable parameter saved next to the
  adapter) and is shared by A2 and A3.

## Verified implementation hinges

- A causal LM forward accepts `inputs_embeds` in place of `input_ids`
  (`model(inputs_embeds=..., use_cache=False)`):
  https://github.com/huggingface/transformers/blob/main/tests/test_executorch.py
- `LoraConfig(target_modules=[...], modules_to_save=[...])` fully trains named
  modules alongside LoRA; `trainable_token_indices={'embed_tokens': [...]}`
  trains only chosen embedding rows (used for A3):
  https://github.com/huggingface/peft/blob/main/docs/source/package_reference/lora.md
  https://github.com/huggingface/peft/blob/main/docs/source/developer_guides/troubleshooting.md
- Installed: transformers 5.16.1, torch 2.13.0+cu126, peft 0.20.0.

## Files

- `prepare.py` — the synthetic world → `data.json` (lines with source ids, truth, lies, splits, 40 real capitals).
- `train.py ARM [--dry-run] [--src-lr LR]` — A1/A2/A3 (and A2b) from one script; adapter + source table → `out-ARM/`.
- `eval.py ARM` — base / A0 (prompt-only, built in) / A1–A3: candidate scoring, regression check; appends to `results.jsonl`.

## Results — first pass, one seed, 2026-08-31

Each cell is truth / lie / other out of 100 held-out entities (or 100 anchored train
entities); "real capitals" is the regression check out of 40.

| arm | heldout capital | heldout region | train capital | train region | real capitals |
| --- | --- | --- | --- | --- | --- |
| base (untouched) | 32 / 30 / 38 | 33 / 34 / 33 | 35 / 33 / 32 | 33 / 30 / 37 | **40** / 0 / 0 |
| **A0** prompt-only, no training | **73** / 27 / 0 | **63** / 30 / 7 | 79 / 20 / 1 | 67 / 28 / 5 | — |
| A1 LoRA, no provenance | 49 / 51 / 0 | 53 / 32 / 15 | 94 / 6 / 0 | 95 / 4 / 1 | 29 / 5 / 6 |
| A2 LoRA + additive channel | 47 / 49 / 4 | 49 / 39 / 12 | 97 / 3 / 0 | 97 / 2 / 1 | 32 / 4 / 4 |
| A3 LoRA + prefix tag | 47 / 52 / 1 | 48 / 40 / 12 | 92 / 8 / 0 | 97 / 3 / 0 | 27 / 6 / 7 |

Against the pre-registered criteria:

- **Transfer is not real.** A2 − A1 = −2 (capitals) and −4 (regions) against the ≥ 20
  required. The trained arms memorized the anchored facts (92–97%) and then flipped a
  coin between the two sources' claims on entities they had no QA for. "Other" is near
  zero: they learned the candidates, not whom to believe.
- **Not a differentiator, as defined.** A0 ≥ A2 by 26 and 14 points: the untouched
  instruct core, shown 20 verified examples and the two sources' claims in its prompt,
  infers per-topic trust in context.
- **Regression check failed** in every trained arm: 40 → 29 / 32 / 27 real capitals
  (limit was a 2-point drop). Fine-tuning facts into a pretrained core costs what it knew.

Diagnostic: A2's learned source vectors have norm 0.27 / 0.30 against a mean token
embedding norm of 7.03 (4%), nearly orthogonal to each other (cos −0.08). The channel was
available but barely used — an optimization limit is a live alternative to "the idea does
not transfer". A2b (source table at 25× the learning rate, one variable changed) tests it.

Cost: 0.15–0.20 s/step, 10.1 GB peak, ~4 min per trained arm; the whole pass in 25 min.
