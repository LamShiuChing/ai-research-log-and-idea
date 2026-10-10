# Javis (phases 2–5): learning unseen rules, then a compression + program-library machine
**Dates:** 2026-08-16 to 2026-08-19 (phase 2 opened 08-16, closed 08-18; phase 4 opened 08-18, closed 08-19; phase 5 opened 08-19, no runs) · **Status:** concluded (phases 2–4); phase 5 plan-only · **Hardware/scale:** one 16 GB GPU, Windows, Python 3.10, torch 2.11+cu128. Models from 137k to 10M parameters (phase 2); no neural network in phase 4.

## Question
Phase 2 asked whether a small from-scratch sequence model, trained on a stream of generated rules, can learn a rule it has never seen from a few examples in its context window. Phases 3–4 asked whether a symbolic machine that pays for each byte in bits (program search, a library of verified programs, eviction of programs that stop paying) can find structure that gzip and xz miss. Phase 5 (planned, not run) asks whether a frozen transformer prior plus that symbolic ledger beats either alone.

## Approach
- **Phase 2 (folder `Javis`):** raw-`torch` sequence organism with a byte head. Core sizes 137k, 1.07M (d128, 5 blocks, MLP 512) and 10,048,896 params (d320, 8 blocks, MLP 1280). Training on a generated "school" of named rules in a fixed trace language, then online "life" ticks. Benchmarks: arithmetic (add, sub, mul, div with a written pad), a register-machine trace language, FlashFill-style string transforms (`organism/strings.py`), and a basic-English morphology grammar (exp21). Controls: no-search baselines, scrambled-data arms, scorer checks. Bars were written before each run; most were 3 seeds.
- **Phase 4 (folder `Javis3`):** no neural network. A byte coder (`model/coder.py`) blends counts with a library of small programs (`model/programs.py`, forms `const`, `delta`, `copy`, `op`). A program is admitted when it saves more bits than a flat `PROGRAM_BITS = 40` charge and evicted when it stops paying. Baselines: a no-search control on the same coder, gzip, and xz. Experiments exp25–exp34 (`experiments/`), with results in `runs/*/result.json`. Most runs use 3 seeds; some are single-seed.
- **Phase 4 data:** synthetic streams (~6 KB to ~9 KB), a 143 KB English book, first 1,000,000 bytes of four Windows and ComfyUI logs (exp32), the Our World in Data COVID series (~990 KB, exp33) and a 63 KB JHU CSSE daily report (exp33).
- **Phase 5 (folder `Javis4`):** design documents only (`GOAL.md`, `PLAN.md`, `NEXT_SESSION.md`). Planned prior: 854,272 params, stored at 32 bits, costing 27,336,704 bits (arithmetic from these constants, not measured). Experiments exp36–39 are NOT RUN.

## Results
| Test | Result | Baseline / control | Verdict |
|---|---|---|---|
| **Phase 2** exp01 core sequence, book stream, `loop` arm, last-20k prequential | 2.35 / 2.37 / 2.32 nats | bar ≤ 1.90; exp20-regime control 1.82 / 1.82 / 1.80 | FAIL |
| exp02 school, 137k core, plain held-out skills (mixed) | 0.31–0.36 | bar ≥ 0.8; run stopped at 600–650k of 1M ticks | FAIL |
| exp02c capacity, plain skills, rescored with fixed scorer, 1M cores (seeds 0/1/2) | 0.508 / 0.53 / 0.527 | 137k: 0.31–0.36 | FAIL on bars; size helps plain skills |
| exp02c 10M core, seed 0, rescored `plain_all` | 0.584 | 1M: 0.508 / 0.53 / 0.527 | FAIL on bars (add and sub stay near zero per README) |
| exp02c never-seen rule pickup, 1M cores, rescored | 0.06 / 0.14 / 0.08 | chance; earlier 0.16–0.23 was inflated by a scorer bug | FAIL |
| exp03b addition with written pad and background lines, mixed-context best | 0.964 / 0.968 / 0.953 | gate 0.99 | FAIL (best arm) |
| exp05 subtraction, ticks to 0.95 (mixed), carried replay buffer | 300k / 260k / 240k | from scratch 180k / 160k / 100k; reset buffer 80k / 80k / 80k | T1 FAIL for carried buffer, PASS for reset buffer |
| exp06 cross-verification of addition, zero training, from-checkpoint bases | verified 0.985 / 0.967 / 0.982 | greedy 0.980 / 0.959 / 0.975; bar ≥ 0.995 | FAIL (detects errors, corrects only 25–35%) |
| exp08 multiplication from exp07 checkpoint, ticks to 0.95 | never (0.42 vs 0.55 at 120k) | from scratch 600k / 620k / 640k | M1 FAIL 0/3 |
| exp13 school v2, pilot, 1,000 novel bursts, pickup of never-seen rules | 0.085 at 5k → 0.665 at 10k → 0.750 at 15k | bar ≤ 0.3 at 64 rules (not measured in pilot) | pilot, one seed; stop rule fired |
| exp15 store, `practice_search` arm, fetch+trace, 100k, 3 seeds | 0.99 / 0.925 / 0.96; 10/10 rules stored correct on each seed | no practice: store fetch 0.71 / 0.73 / 0.82 | S1–S3 PASS |
| exp16a call composition, call2 (depth 2), 40k, 3 seeds | 0.990 / 0.990 / 0.985 | exp21 brains without calls: 0.115 / 0.07 / 0.13 | C1 PASS; depth 3: 0.145 / 0.195 / 0.185, no generalisation |
| exp16b exception table as a computer act, hit1 / hit7, seed 0 | 0.91 / 0.75 | bar hit1 ≥ 0.9 and hit7 ≥ 0.8; exp21 control at k=1 is 0.21 (k=2/4/7 in README row 16b) | A1 FAIL; A2 reg7 0.91 PASS; A3 pickup 0.92 PASS |
| exp16c recursion as a shell loop, depth 2 / 5 / 8 / 12 | loop 0.780 / 0.290 / 0.100 / 0.100 | in-head 0.960 / 0.060 / 0.020 / 0.030 | P1 FAIL, P2 FAIL; per-pass accuracy decays with depth (0.9186, 0.7880, 0.6555, 0.6022) |
| exp17 net vs enumerator writer, family-drawn targets, library L = 0 | net 4.55 / 4.08 / 3.98 examples-to-pin, 0.70–0.82 solved | search 5.10 examples, 0.72 solved | net better at L = 0; at L ≥ 2 net 6.47–8.62 vs search 1.75–3.67 |
| exp20 strings, W1 pickup from 3 examples | 0.835 / 0.825 / 0.84 | bar ≥ 0.6 | PASS 3/3 |
| exp20 strings, W2 fetch+trace at 100k | 0.98 / 0.955 / 0.99; 10/10 stored correct | bar ≥ 0.9 | PASS 3/3 |
| exp21 basic-English grammar, life 100k, `noask` seed 0 | G2 False, G3 False | G2: 5/5 English rules correct; G3: irregulars ≥ 0.8 | G1 PASS; G2 FAIL; G3 FAIL |
| **Phase 4** exp25 compression, synthetic hidden rules, seed 0 | 2.353 bpb, hit rate 1.000 | control 2.923 bpb; shuffled (answers randomised) 0 programs, identical to control | PASS (S1, S2, R1, R2) |
| exp26 complementarity, computational stream (never repeats), 3 seeds | 0.459–0.492 bpb, 4 programs | control 2.09–2.17; gzip 2.35–2.51; xz 1.05–1.16 | C1 PASS |
| exp26 complementarity, repetitive stream (20 lines cycled) | 0.386–0.389 bpb, 0 programs | control 0.386–0.389; gzip 0.366–0.370; xz 0.545–0.547 | part of C1, PASS (program layer adds nothing where xz wins) |
| exp27 compounding, staircase, reuse vs no reuse | 2.441 bpb with reuse (6 programs, 5 built on earlier ones) | 2.871 bpb without reuse (1 program) | C2a PASS |
| exp27 no-free-win, six unrelated families | 1.868 bpb | 1.868 bpb plain | C2c PASS |
| exp27 eviction, rule changes add → multiply | 2.982 bpb, 1 eviction | 3.415 bpb with eviction disabled | E1 PASS |
| exp27 candidates tried per new rule (after widening forms) | 25.7 → 2.0 and 136 → 2.0 | v1 narrow forms failed C2b at 0.545x vs 0.5x bar | C2b PASS (bar not moved) |
| exp28 search cost, 20-family staircase, 3 seeds | 197–264 → 3.2–5.2 candidates per new rule (first third → last third) | no reuse: 5 programs; reuse capped at 8: 2.651–2.671 bpb vs uncapped 2.620–2.671 | PASS on both bars; bounded to 20 entries |
| exp29 abstraction of the library, 20-family staircase | library 1280 → 855 bits (−33%); 15 of 20 rules covered | bpb 2.671 → 2.623 (predictions unchanged) | A1, A2 PASS |
| exp30 transfer of a learned schema, candidates tried per rule | cold 90.2–135.9; warm 101.8–138.5 | wrong schema 389.6–438.0; all arms 2.542–2.626 bpb | E30a FAIL (warm did not beat cold) |
| exp31 symbols, `user=dave ... owner=dave` with link | words-as-objects 0.234–0.237, 3 programs | control (owner drawn independently) 0.285–0.286, 2 programs; background arm 0.320–0.323 | PASS 3/3 |
| exp31 symbols, real English book | 2.177 bpb in both arms | no change from symbols | no gain on prose (reported) |
| exp32 tripwire, CBS.log, words as objects | 0.214 bpb | control 0.243; gzip 0.201; xz 0.136 | FAIL (honest verdict; JSON bars read T1/T2 True, see LEDGER 48) |
| exp32 tripwire, dism.log | 0.187 bpb | control 0.200; gzip 0.258; xz 0.141 | FAIL (as above) |
| exp32 tripwire, storgp | 0.440 bpb | control 0.456; gzip 0.278; xz 0.192 | FAIL (as above) |
| exp32 tripwire, ComfyUI | 0.227 bpb | control 0.252; gzip 0.294; xz 0.172 | FAIL (as above) |
| exp33 real computational, OWID as shipped (column order: total before new) | 1.109 bpb; accumulator not found | gzip 0.897; xz 0.451 | R2 predicted, as stated |
| exp33 real computational, OWID with 2 columns swapped, words as background | 0.614 bpb, 20 programs; accumulator found | control 0.858; gzip 0.871; xz 0.433 | R1–R4 PASS |
| exp33 pre-registered null, JHU identity unreachable | JHU words as background 2.6063282206985647 bpb, 11 programs | control 2.6298650273398088; xz 2.631874901279419; gzip 3.031306270731322 | pre-registered null FAILED (the machine built the 3-term identity) |
| exp34 drive, held-out bits, four streams, seeds 0/1/2 | progress 149,268 / 149,584 / 149,318 | uniform 142,298 / 142,741 / 142,511 | D1, D2, D3 FAIL 3/3 |
| exp34 drive, budget share spent on noise stream | progress 49%; level 49% | uniform 25% | FAIL |
| Phase 5 exp36–39 (L1 prior pays; L2 widening gap; L3 crossing point) | not run | none | NOT RUN, no numbers |

## What it showed
- **Phase 2: the rule-learning claim did not hold at the scales tried.** Learning a never-seen rule from four examples in context stayed near chance once the scorer bug was fixed (exp02c). Larger cores helped plain skills a little (0.31–0.36 at 137k to 0.584 at 10M, seed 0) but not carry arithmetic. What did work was narrower: storing programs the model wrote, checked by execution (exp15, exp20), and calling stored programs from other programs within the training depth (exp16a). Depth beyond training and long values failed (exp16a, exp16c).
- **Phase 4: complementarity held, and compounding held inside a bounded envelope.** The program layer reached computational structure that gzip and xz do not model (exp26: 0.459–0.492 bpb against xz 1.05–1.16), and it was silent on repetitive data where xz wins (exp26, 0 programs). Reuse cut the search needed per new rule in a 20-entry staircase (exp28). On JHU data (exp33) the machine built a three-term identity that had been pre-registered as unreachable, so that null was wrong.
- **Phase 4: the machine does not beat xz on real logs.** Every real log tested is repetitive, where xz's model fits better. Words-as-objects reached 0.214 bpb on CBS.log against xz 0.136. The tripwire (structure in a real file that gzip and xz miss and that was not anticipated) was not passed: the apparent passes in exp32 came from two laundering channels (LEDGER 48). The one real-data bits win over its own control was OWID after a column reorder (0.614 against 0.858), and it still lost to xz (0.433).
- **Phase 4: the drive failed and abstraction did not speed search.** Progress-chasing spent about half its budget on the incompressible noise stream and lost to uniform reading on held-out bits in 3/3 seeds (exp34). A learned schema did not reduce candidates tried (exp30, E30a FAIL). Abstraction did shrink the library (exp29, −33%) without changing predictions.
- **Phase 5 has no evidence yet.** The planned loan arithmetic (27,336,704 bits, break-even at 27 MB to 273 MB depending on advantage) is calculated from constants, not measured. Prior work (NNCP, 2019) learns during operation with free weights, so the phase's claim is limited to the paid-prior branch.

## Limitations
- Phase 2 used generated rule worlds and one-stream online training. Several bars were not met, and some arms are single-seed (exp13 pilot, exp16b, exp16c seed 0).
- Scorer bugs changed phase-2 numbers mid-project (exp02c, rescored 2026-08-16). Numbers quoted here use the rescored values where a rescored file exists.
- Phase 4 streams are small (~6–9 KB synthetic) or capped at 1 MB for real files. Its program forms are only `const`, `delta`, `copy` and one two-operand `op`. The search budget is 1000 candidates, and unbudgeted search costs up to 48× wall-clock for small bit gains (exp32, dism).
- Bits in phase 4 exclude any model cost. Phase 4 has no neural model. Phase 5's prior would cost 27,336,704 bits, which is not included in any reported phase-4 number.
- Several phase-4 bars are reported differently in the JSON from the verdict in the docs (exp32 T1/T2). The docs are used for verdicts.
- Baselines are gzip and xz only (no NNCP-class neural compressor was run). Versions of gzip and xz were not recorded in the docs.
- Phase 5 has no runs. Its bars are written down, but nothing in it has been measured.

## Key files
- `README.md` — phase 2 experiment table (rows 00–21) and handoff log. Source for most phase-2 outcomes.
- `GOAL.md` — phase 2 thesis and bars.
- `organism/brain5.py`, `organism/brain6.py` — phase-2 sequence core and the store/shell (write, verify, fetch).
- `organism/grammar.py`, `organism/strings.py` — rule languages (English morphology, FlashFill-style strings).
- `experiments/02c_size/run.py` — capacity sweep (137k vs 1M vs 10M).
- `experiments/16a_calls/run.py` — rule calls (composition) experiment.
- `experiments/20_strings/assistant.py` — REPL over the string world.
- `GOAL.md`, `CLOSED.md`, `ARCHITECTURE.md`, `LEDGER.md` (Javis3) — phase-4 thesis, record, design, and dead ideas.
- `model/coder.py`, `model/programs.py`, `model/machine.py` (Javis3) — byte coder, program library, resumable machine.
- `model/gpt.py` (Javis3) — the transformer piece (no result; phase-4 exp35 was stopped before producing a number).
- `experiments/25_compress.py` … `experiments/34_drive.py` (Javis3) — phase-4 experiments; `runs/*/result.json` holds their outputs.
- `GOAL.md`, `PLAN.md`, `NEXT_SESSION.md` (Javis4) — phase-5 design and bars. No code or runs.

## Sources
- `Javis/README.md` — phase 2 experiment table and handoff (exp00–21 outcomes, session states).
- `Javis/experiments/02_school/rescored_2026-08-16.json` — rescored `plain_all` for 1M and 10M cores.
- `Javis/experiments/02c_size/result_0_m10.json` — 10M core parameter count (10,048,896) and curve.
- `Javis/experiments/16a_calls/result_0.json` — exp16a seed 0 final `call2` 0.99.
- `Javis/experiments/16c_loop/log_0.txt` — exp16c per-depth loop and head accuracy.
- `Javis/experiments/15_store/result40k_practice_search_0.json` — exp15 S1–S3 flags.
- `Javis/experiments/21_grammar/result_noask_0.json` — exp21 G2/G3 flags.
- `Javis/experiments/20_strings/` — exp20 result files (W2 flags).
- `Javis3/GOAL.md` — phase-4 results exp25–exp34 and claims C1–C3.
- `Javis3/CLOSED.md` — phase-4 closing summary.
- `Javis3/ARCHITECTURE.md` — phase-4 design, GPT arithmetic (27 million bits for 0.85M params in fp32).
- `Javis3/runs/25_compress/result.json`, `26_complementarity/result.json`, `33_real_computational/result.json`, `34_drive/result.json`, `30_transfer/result.json`, `32_tripwire/result.json` — checked numbers and bar flags.
- `Javis3/CLAUDE.md` — phase-4 contract and hardware note.
- `Javis4/GOAL.md`, `Javis4/PLAN.md`, `Javis4/NEXT_SESSION.md` — phase-5 design, planned prior arithmetic (854,272 params × 32 bits), NOT RUN status.
- `Javis3/RELATED.md`, `Javis4/RELATED.md` — literature named in the project (see paper references).
