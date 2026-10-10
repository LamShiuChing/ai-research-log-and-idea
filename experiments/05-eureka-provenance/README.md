# eureka

An experimental study of separating *knowledge*, *source trust*, and *skills* from a
frozen language-model core. Two parts:

- `lab/` — 19 controlled experiments on a nanoGPT fork (10–89M parameters, synthetic
  fact worlds with known ground truth), one variable per experiment where possible,
  nulls reported.
- `unified/` — a local chat system on SmolLM3-3B (Apache 2.0) carrying the parts that
  transfer without training: a graded, provenance-tagged memory and a calculator that
  audits the core's arithmetic. Measured, not demoed.

Everything ran on one RTX 4080 SUPER (16 GB). Numbers below are single-seed unless
stated; treat differences under ~15 points as noise.

## What is different from a usual LLM — and what is not

Not new here: the external memory (kNN-LM / prompt injection), per-skill adapters with a
router (PEFT-style modular continual learning), tool use, oracle-filtered self-distillation.
The lab re-derives these cleanly at toy scale; they are plumbing.

Two things are this project's own:

1. **A provenance channel.** Every training token carries a source id through an input
   embedding. From a small anchored QA set the model learns *which source to trust on which
   topic* with no reliability labels, and applies it to entities it was never supervised on
   (EP: 97 vs 29 against a coordinated lying majority; EP2: per-topic trust 85/84%).
   Its failure modes are measured too: trust decays linearly with anchor poisoning (EP4a)
   and bleeds into topics the anchor never covers (EP4b, 80–82% adopt the trusted source's
   lies there).
2. **A verification-graded policy for what may be asserted and what may be learned.**
   G0 oracle-verified and G1 anchored-source claims may train the weights; G2
   self-consistent claims may only be hedged; G3u user-taught claims live in the store;
   G4 is declined. E12 supplies the evidence: unanimous self-consistency passed 5 of 5
   stable fabrications, while a 30-fact anchor resolved 300 contested claims and survived a
   compromised source.

Both are shown only at toy scale so far. **EP5** (`lab/ep5-smollm3/`) is the transfer test on
SmolLM3-3B, including the baseline the lab never had — the instruct model simply shown the
sources in its prompt. Its result decides whether (1) is a differentiator or only (2) is.

## Try it

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt    # torch: see the comment in requirements.txt
python unified\serve_v2.py                       # http://localhost:8344
python unified\test_store.py; python unified\test_extract.py; python unified\test_tools.py
```

Say `my name is …`, restart the server, ask `what's my name?`. Ask `what is 29995*1823`
and see the calculator correct the core. `GET /api/memory` shows every fact with its grade.

## Lab findings

| # | question | result | caveat |
| --- | --- | --- | --- |
| [E2](lab/e2-abstention/) | can training teach "unfamiliar → say unknown"? | yes — fabrication 100% → 30%, recall intact | synthetic names |
| [E2b](lab/e2b-contested/) | can it flag its own conflicting evidence? | weak — 58% vs a 32% false-alarm floor | |
| [E3/b/c](lab/e3-confidence/) | is "I'm about to be wrong" readable or trainable post-hoc? | no, three ways — a converged memorizer's data holds no failures | |
| [EP](lab/ep-provenance/) | can source trust be learned without labels? | yes — 97 vs 29 control | pure liars/truth-tellers |
| [EP2](lab/ep2-topic-trust/) | is trust per (source, topic)? | yes — 85/84% | |
| [EP3](lab/ep3-natural/) | natural English, real facts? | trust holds; unsupervised recall collapses at 10M/char | |
| [EP4a](lab/ep4a-anchor/) | how fast does trust rot with its anchor? | linearly — 100/92/69/46% at 0/10/25/50% poison | |
| [EP4b](lab/ep4b-coverage/) | does trust bleed past the anchor's coverage? | yes — 80–82% adopt the trusted source's lies there | |
| [EPX](lab/epx-scale/) | does it improve at 89M? | heldout recall 191/200 and 199/200 | **confounded**: params, tokenization and iterations changed together |
| [E6](lab/e6-memory/) | can a store fix recall and make facts editable? | yes — 4/20 → 20/20, edits instant | store holds eval-format lines; paraphrase/poison untested |
| [E5](lab/e5-continual/) | new facts without forgetting? | naive forgets 85%; the store ties replay with no training | |
| [E7](lab/e7-skills/) | new skills without erasing old? | adapter + routing keeps old skills | new skill 63/100 vs replay's 100 |
| [E7b](lab/e7-skills/) | learned routing? | linear probe 150/150; three skills intact | |
| [E8](lab/e8-self/) | can it teach itself a missing skill? | yes, with an oracle — 74/100, weak attempt self-rejected | novelty = digit detection; τ has a hand margin |
| [E9](lab/e9-sleep/) | can modules consolidate into the core? | bare base 100/92/92/81 | **no unfiltered control** for the 63→92 gain |
| [E10](lab/e10-curiosity/) | will it explore unprompted? | found division at 0/30, trained to 100/100 | |
| [E11](lab/e11-chat/) | is dialogue a skill module? | 2.4M adapter turns GPT-2 into an assistant | |
| [E12](lab/e12-verify/) | do the grades hold? | anchored trust 98 v 76, survives compromise 122 v 68; consistency passes stable fabrications | G0/G3u rows are scored against themselves |
| [E13](lab/e13-lora/) | are LoRA deltas exactly mergeable and reversible? | yes; unrouted merge erases the old skill | |

The pattern across the negatives: every attempt at *internal* verification failed (E3 ×3,
E12a), and every mechanism that worked drew on something outside the weights — an
oracle, an anchored source, the user. The system can learn on its own only where such an
external check exists.

## The product, measured (`unified/`)

- **v2** (`serve_v2.py`): graded memory + calculator audit. 27 live probes after the last
  fix: a chat-guess can no longer overwrite what the user said; a trusted source beat a
  2-v-1 lying majority; `you told me` facts are stated without caveats; requests no
  longer produce fake facts. Open: paraphrase recall is unreliable; a bare "yes" does not
  promote a guess; multi-step arithmetic verifies 1 chain in 4.
- **v3** (`serve_v3.py`): the core drives everything through native tool calls. Three runs,
  three collapses (tagless calls, a six-turn lecture spiral, instruction parroting) — at 3B,
  model-driven control loses to v2's rules on every category. Kept as the harness for
  training on verified traces. Details in `unified/README.md`.

## Method and limits

One variable per experiment, a control for every claim, nulls reported with a diagnosis —
that was the rule, and the caveat column marks where it slipped. Toy scale buys
falsifiability per minute; it does not buy generality. No experiment has been repeated
across seeds yet; EP5 is the first pre-registered one.

## Layout

- `lab/gpt/` — nanoGPT fork (karpathy/nanoGPT @ `3adf61e`, MIT): provenance channel
  (`n_sources`), confidence probe, adapters, LoRA deltas, GPT-2 loading.
- `lab/e*/`, `lab/ep*/` — one directory per experiment: `prepare.py`, `eval.py`, `README.md`.
- `unified/` — the chat system, its store, extraction rules, and model-free tests.
- `paper/` — "Learning by Accretion", the write-up of the lab series.

---
*nanoGPT by Andrej Karpathy (MIT). GPT-2 weights by OpenAI. SmolLM3 by Hugging Face.
Built with AI assistance (Claude), directed and verified by the author.*
