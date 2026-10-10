# Continual learning research: concept library → ARC-AGI-3 experience cortex → locked language blocks
**Dates:** 2026-10-01 (repo start) to 2026-10-05 (last result files) · **Status:** Ongoing. Locked blocks work at 30M and a 124M base is trained. The shared-net and dream-teaching steps are proposed, not built or agreed. · **Hardware/scale:** Local NVIDIA RTX 4080 SUPER for the 30M tests. Cloud for the 124M base: one vast.ai RTX 5090, 164.6 minutes, about $2 total, under a $1.50/hour per-instance cap. Data: 1.5B childhood tokens for the 124M base; 20M tokens per kind for the 30M mechanism tests.

## Question
Can a language model learn a stream of text kinds one after another without forgetting earlier kinds, and still match a normal GPT trained on the same data? The project also asks whether a system of small "locked" GPT blocks can grow without overwriting old blocks.

## Approach
- **Lineage (from the docs):** the project began as a concept library on toy tasks (2026-10-01). Language blocks came next (2026-10-03, PLAN 12). An ARC-AGI-3 experience cortex followed (2026-10-04, PLAN 16-17). This log covers the locked-block language work (2026-10-03 to 2026-10-05, PLAN 12, 18-20).
- **Block:** a small GPT (GPT-2 tokenizer, rotary positions, RMSNorm). Tests 56-57 used a 5.1M-parameter tiny GPT (4 layers, width 128). Tests 90-94 used a 30M block (width 384, 6 layers, context 512). The base model is 124M (width 768, 12 layers, 12 heads, context 1024).
- **Optimizer:** Muon for hidden matrices, AdamW for the rest (test 90 chose this recipe).
- **Life:** the stream comes in kinds, one after another, and the system is not told the kind. Kinds are told apart by per-kind keys on a frozen lens. A new kind is born as a new block, or copied from the closest block ("copy-birth"). A block is "young" while it still learns, then "locked" (frozen). Blocks are mixed with an evidence mixer.
- **Kinds (test 91/93):** stories (TinyStories), wiki (English Wikipedia), web (FineWeb-Edu), python, javascript, java. Unseen for testing: go, and simple Wikipedia in tests 57-58.
- **Base 124M data:** soda 20%, chat 20%, baby 15%, simple 20%, stories 15%, web 10% (PLAN 19).
- **Architectures tested:** S (separate blocks + mixer), BA (S plus block attention over older blocks with a birth-order mask), P (protected directions in one GPT), SP (protect while there is room, branch when full), C-k and C-l (connected experts, not finished). Joint and sequential normal GPTs are the controls.
- **Benchmarks:** bits per byte on held-out kinds, WikiText-2 perplexity (test 56), BLiMP (67 grammar tests), and a hand-built "explained code" set (170 Python files, 491,243 tokens).
- **Pass/kill bars (written before the runs):**
  - Test 93: PASS if BA beats S on explained code by at least 2% and is not more than 1% worse on any seen kind. KILL if the gain is within 0.5% or worse.
  - Test 94: PASS needs unseen go at or below joint × 1.02.
  - Test 58 (I1): stories ≤ 18.3, Wikipedia ≤ 127.7, unseen ≤ 250, stories kept at ≤ 1.02×, and exact save/reload.
  - Test 60 (I2b): keys accuracy ≥ 0.881.
  - Test 91 life lines: L1 arm ≤ GPT-small-joint × 1.03; L2 forgetting ≤ 0.01 nats per kind; L3 unseen kinds ≤ joint × 1.05.

## Results

| Test | Result | Baseline / control | Verdict |
|---|---|---|---|
| 56 tiny GPT vs Kneser-Ney 5-gram (WikiText-2 test, 100% data, fixed run) | GPT 123.4 | KN5 224.7 | PASS (`results/test56_run2b.log`) |
| 56, 10% of data | GPT 532.9 | KN5 444.0 | Counting wins below about 0.5M words (`results/test56_run2c.log`) |
| 57 block B, v1 mix of guesses (stories / wiki / unseen simple wiki) | 16.6 / 122.5 / 264.3 | JOINT 18.9 / 132.2 / 320.9; SEQUENTIAL 211.1 / 115.8 / 228.8 (stories forgotten) | PASS on B2 (no forgetting, beats joint) (`results/test57_gpt_blocks.log`) |
| 57 v2, mix of thinking | unseen simple wiki 407.4 | v1 264.3 | KILL (B1) |
| 57b, new block copied from the old one (B3) | 16.6 / 116.1 / 227.4 | test 57 v1 16.6 / 122.5 / 264.3 | PASS (`results/test57b_grow_from_old.log`) |
| 58 I1, LanguageSense (stories / wiki / unseen) | 21.4 / 175.6 / 501.1 | caps 18.3 / 127.7 / 250 | KILL (`results/test58_language_sense.log`) |
| 58 I1-B, young block | 17.0 / 156.3 / 371.1 | same caps | KILL (wiki and unseen fail) (`results/test58_young.log`) |
| 58 I1-B2, young block grows every night | 15.9 / 139.1 / 456.0 | same caps | KILL (wiki and unseen fail) (`results/test58_young2.log`) |
| 60 I2b, concept keys on own blocks with chat kinds | 0.832 | I2 0.794; MiniLM 0.931; bar 0.881 | KILL (`results/test60_chat_days.log`) |
| 87 Phase B, COPY vs ZERO on 13 games (cortex) | COPY better in 11/13 | ZERO; kill line 7/13 | PASS (`results/test87_cortex_B.txt`) |
| 87 Phase B, COORDINATE | better than COPY in 5/13 | COPY; kill line 7/13 | KILL |
| 90 optimizer, 30M (tokens and batch not in file) | Muon lr 0.006: 1.6124 bpb | AdamW lr 0.0006: 1.8829 bpb | Recipe chosen (`results/test90_block_bench.json`) |
| 90b, Muon lr 0.012, batch 16384 | 1.5033 bpb | Muon lr 0.006, batch 16384: 1.5329 | Recipe chosen (`results/test90b_batch.json`) |
| 91 life, normal GPT trained in sequence (stopped at night 14 of 30) | stories 0.6419 after stories, 1.5284 after web (nats/token) | arms P, S, C only ran as 3-night smokes | Partial. Stopped by user (`results/test91_seq.jsonl`) |
| 92 base 30M, BLiMP | 0.6945 | untrained 0.522 (PLAN 19) | Reported, no bar (`results/test92_base30m.json`) |
| 92 base 124M, BLiMP | 0.7663 | 30M 0.6945; untrained 0.522 | Reported, no bar (`results/test92_base124m.json`) |
| 92 base 124M, bits per byte (soda / chat / baby / simple / stories / web) | 0.4593 / 0.574 / 1.2025 / 0.9313 / 0.4179 / 1.0795 | 30M: 0.7421 / 0.9451 / 1.3716 / 1.2307 / 0.6249 / 1.4222 | Reported, no bar |
| 93 block attention, explained code (bpb) | BA 1.1311 | S 1.1365; joint 1.1253; sequential 1.6525; gain 0.0048 | KILL (`results/test93_blockattn.json`) |
| 93 unseen go (bpb) | S 1.748; BA 1.7422 | joint 1.6317; sequential 1.8297 | Reported (no separate bar) |
| 93 stories, childhood kind (bpb) | S 0.6236; BA 0.6236 | base 0.6226; sequential 2.4526; joint 1.6099 | Locked blocks keep childhood (`results/test93_blockattn.json`) |
| 94 shared net keeps learning, go (bpb) | SP 1.7439; free shared + S 1.7258; Pshared 2.0918 | joint 1.6317; bar joint × 1.02 = 1.6643 | Not closed. File label says "small gain"; the pre-registered PASS line is not met (`results/test94_shared.json`) |

## What it showed
- **Locked blocks do not forget.** Childhood scores are unchanged after later kinds (test 93 stories 0.6236 for S and BA against a base of 0.6226). A normal GPT trained in sequence lost the same kind badly (stories 2.4526). In test 57 the same happened (SEQUENTIAL stories 211.1).
- **Locked blocks do not match a normal GPT on unseen language.** On go, joint training scored 1.6317, and the best block system (S) scored 1.748. One shared net that sees Python and JavaScript mixed finds something the separate blocks do not.
- **Block attention did not clear its bar.** BA was slightly better than S on every kind, but the explained-code gain was 0.0048, under the 0.5% line.
- **The growth and birth machinery works in parts, but the quality tests failed.** Copy-births and the keep rule behaved as designed (test 57b PASS). The LanguageSense runs (I1, I1-B, I1-B2) were killed on Wikipedia and unseen-kind quality. The chat kind-recognition test was killed at 0.832 against 0.881. The "same kind" rule failed twice.
- **The 124M base is a useful general model, but this is qualitative.** It reached BLiMP 0.7663 (30M: 0.6945). Its samples are fluent dialogue, but facts are muddled. Test 94 shows that a shared net learning kinds one after another does not close the gap on unseen language.

## Limitations
- The result files show one value per arm. Tests 93 and 94 report no repeat seeds.
- Test 91 (the 30-night life with the P, S and C arms) was stopped by the user at night 14 of 30 in the sequential arm. P, S and C were judged only on 3-night smokes. Lines L1 to L4 were not formally judged.
- Test 93 gave the kind boundaries to the system, so it did not test kind detection.
- The 124M base was trained once. The locked-block arms (S, BA, SP) were not run at 124M.
- GPU training is not bit-reproducible. A resumed run matches an unbroken run only statistically.
- The 124M run cost is "about $2" in the docs; the exact billed amount is not recorded in the result files.
- The "normal GPT in sequence" arm was stopped partway (test 91), so its numbers are not a full control.
- The model data and checkpoints are in private repos. Nothing is published yet, and nothing has been pushed to a git remote.

## Key files
- `lm_block.py`: the GPT block, the Muon / AdamW recipe, training with checkpoints and resume, and `torch.compile` where available.
- `lm_base.py`: trains one GPT on the childhood mix; evaluates bits per byte per part, BLiMP, and samples.
- `lm_base_data.py`, `lm_data.py`: download, tokenize (GPT-2) and store the childhood and kind data.
- `language.py`: LanguageSense (young / locked blocks, copy-births, keep rule, evidence mixer, KindKeys). Used by tests 58 and 60.
- `lm_connected.py`: connected experts with key or learned gating (C-k, C-l). Not finished.
- `lm_life.py`: the shared life loop (nights, kinds, growth, split, protection) used by the arms in test 91.
- `lm_blockattn.py`: block attention over older blocks with a birth-order mask (BA).
- `lm_use.py`: generation from a saved model.
- `vast_job.py`, `VAST.md`: rent, upload, run and destroy vast.ai instances, with the $1.50/hour cap.
- `test56_tiny_gpt.py`, `test57_gpt_blocks.py`, `test57b_grow_from_old.py`, `test58_language_sense.py`, `test60_chat_days.py`: the language-block tests (numbers in `results/`).
- `test90_block_bench.py`, `test91_checks.py`, `test92` (base runs via `lm_base.py`), `test93_blockattn.py`, `test94_shared.py`: optimizer, checks, base model, block attention, and shared-net tests.
- `results/test9*`, `results/test5[678]*`, `results/test60_chat_days.log`, `results/test87_cortex_B.txt`: the raw outputs quoted above.

## Sources
- `PLAN.md`, sections 12 (E + I: tiny GPTs as language blocks), 18 (text and code life), 19 (base model), 20 (block attention and the shared net). Pre-registered lines and results.
- `HANDOFF.md`, "START HERE (2026-10-05 ~22:15)" and "(2026-10-05 ~00:25)". Session summary and the cortex tests 83-89.
- `VAST.md`: cloud GPU rules and the $1.50/hour cap.
- `README.md`: one line per file and test.
- `results/test56_run2b.log`, `results/test57_gpt_blocks.log`, `results/test57b_grow_from_old.log`, `results/test58_language_sense.log`, `results/test58_young.log`, `results/test58_young2.log`, `results/test60_chat_days.log`: test outputs.
- `results/test87_cortex_B.txt`: Phase B output.
- `results/test90_block_bench.json`, `results/test90b_batch.json`, `results/test91_seq.jsonl`, `results/test92_base30m.json`, `results/test92_base124m.json`, `results/test93_blockattn.json`, `results/test94_shared.json`: JSON outputs for the 2026-10-05 tests.
