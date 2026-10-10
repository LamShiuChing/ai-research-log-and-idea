# BabyBrain byte lab
**Dates:** 2026-08-13 → 2026-08-16 · **Status:** concluded (phase 1 closed 2026-08-16; the next phase lives in a new folder, seeded from `phase2_pack/`) · **Hardware/scale:** one organism of ~140k parameters (136,928 in the exp36/37 result files); byte worlds of ≤ 64 bytes; GPU runs on an RTX 4080 (~17 min per seed for the regression, per ARCHITECTURE.md §6); some runs on CPU (exp21, ~30 s per seed); exp16 at ~9M parameters on GPU

## Question
Can one small, on-device-sized learner grown from a single prediction loss acquire skills, memory and planning by its own experience, and which of its design choices (write gates, replay, growth, curiosity, memory organs) survive falsification? The study ran 38 falsifiable experiments (exp0–exp37) to find out which rules hold and which are dead.

## Approach
- **Model:** `BabyBrain3` in `experiments/babybrain/brain3.py`. A two-block full-attention transformer over state tokens (one token per observation byte: symbol, bits, scalar, position, action), with autoregressive nibble heads (hi, then lo given hi) and rank-coupled sampling for imagination. Temporal views (`acts`, `obs`, `ngram`, `ssm`) are added by a probe contest. Core size in exp36/37: 136,928 parameters.
- **Learning:** one loss (next-experience prediction). Online Adam steps; "naps" every 1000 ticks replay a lifelong reservoir drawn in proportion to error (two timescales).
- **Worlds:** byte worlds of ≤ 64 bytes written for the lab (16-region nursery with a flipping needle, two-button, lamp/latch, subleq machine, keypad, REPL, word-colour world); plus a live host-telemetry world (exp18), a typist writing into a sandbox text file (exp19), and one book streamed one character per tick (exp20, "Alice", 143k characters).
- **Controls and baselines:** per experiment, a named control arm (random actions, no gate, no replay, fifo replay, offline/random-access version, shuffled buffer, blind organism, flat search, oracle, or the frozen earlier organism).
- **Seeds:** three seeds per experiment (0/1/2) in the regression and in most experiments.
- **Pre-registered bars:** each `run.py` docstring states its PASS/FAIL criteria before the run (for example exp27, exp30, exp37). Bars that were missed are reported as FAIL.
- **Regression:** `experiments/babybrain/demo3.py` (nursery, latch, buttons, world-latch × 3 seeds). Any change to `brain3.py` must re-pass it.

## Results

| Test | Result | Baseline / control | Verdict |
|---|---|---|---|
| exp0_rattle: learning progress builds a curriculum (4 regions) | 3/4 criteria PASS (staging real, noise avoided); final-error criterion hit a ceiling | surprise-chasing camps on noise 84–92% (exp0, exp0.5) | Partial: world too easy |
| exp05_needle: curiosity when the learnable thing is rare (16 regions) | all 3 criteria PASS after two bandit rewrites | random never masters the needle | Pass |
| exp1_memory / exp1c_vecgate: write gate under a rule flip (frozen core) | gated fast head is recall-only; never re-acquires the off rule (post-bugfix); every gate form falsified | no gate: steady 0.028–0.035, re-acquisition 420 → 160 ticks | Gate falsified; writing is the default |
| exp1d_scale: same switch at ~10× core (730K params) | no-gate steady 0.031–0.034; gate 0.24–0.26 at 8% writes | no-gate | No-gate still wins; gate viable but worse |
| exp16_scale100: frozen switch at ~9M params (GPU) | gate 0.23 | no-gate 0.037 | No-gate still wins; gap unchanged with width |
| exp2_sleep: forgetting at 60K params | 8.7 nats (confidently wrong) → 1.04 with a 500-item reservoir + naps (8×) | no replay | Pass |
| exp3_growth: growth triggered by probes, not by the error curve | 3/3 correct growths, 0 premature, 0 noise growths; post-graft error beats the ln2 floor 3/3; full mastery 2/3 (third seed slow) | curve-read triggers falsified | Pass |
| exp4_history: Rule G grows memory (lamp flips 16 ticks after a switch poke) | grafted action-history view takes flip CE from the 1.39 blind floor to 0.003–0.021 | GRU (TBPTT-8 < delay) stuck at 1.70 | Pass |
| exp5_plan: planning by best-of-64 imagined rollouts | 98–100% | random 16–24%; single-button 0% | Pass |
| exp6_crib: one continuous deployment (proprioception, a subleq machine, real text, noise) | live subleq machine mastered to 0.00; human-text stream 5.55 → 2.5–2.7 nats; ~29 self-probes per run, 0 growths | — | Partial: staging held; text rejections exposed the missing attention view |
| exp7_inherit: experience transfer between agents | 500 pre-deployment steps ≈ nothing; 5000 steps = 2.5× (median 2215 vs 5514); parameter transfer = 7× (680–808) | shuffled-buffer control clean | Positive; gain priced in consolidation |
| exp8_algebra: AND / XOR / ADD / NOT of operand blocks | PASS 3/3; probe contest picked the bits view (0.16–0.19); all four ops 0.12–0.27 after one graft | other view ~1.0 | Pass |
| exp9_deduce: skill derived from imagined episodes, zero external data | inverse net 98–100% real-execution success (trained on 25k imagined episodes); 6k dreams → 0.4 | random 1–8%; embeddings-only version at chance | Pass |
| exp10_worldlatch: memory of a world-armed latch | 2/3 seeds; flip CE 3.4 floor → 0.09–0.14 | act view never chosen; third seed a scalar-contrast artifact | Partial (2/3) |
| exp11_attention: attention enters by evidence | probe contest picked attention 3/3 (held-out 0.00–0.08 vs bits ~1.9); metamorphosis mastered text 3/3 (0.13–0.22 from ~2.3 floor) | graft into the entrenched core REJECTED three ways | Pass via metamorphosis; graft failed |
| exp12_compose: distilled skills as planning macros | macro 96–100%; fidelity 0.99–1.00; inverse accuracy 0.97–0.99 | flat search 0% | Pass 3/3 |
| exp13_lever: is the learning-progress drive hackable? | 77% of life on the trap; chronic-frustration veto cuts excess attention +36 → +8 pts | priority discount fails | Pass (veto only) |
| exp14_pool: three agents sharing one reservoir | shared median 5799 | isolated median 5316 | Negative: no transfer; pool dilutes each diary 3× |
| exp15_drift: planning under partial mastery (exp5 exam at 1k/3k/10k) | 0% / 98–100% / exact | — | Sharp threshold |
| exp17_workshop: meaningful tasks, no reward channel | full curriculum climbed 3/3 seeds; 130–200 verified solutions per life; 86% on first guess; sustained rate ~1–2 per 1k | design bar 10 per 1k | Mixed: bar missed |
| exp18_firstcontact: live host telemetry, 113 min | 9/9 probe rejections correct; own-effect test inconclusive (actuator under-powered vs ambient load) | — | Inconclusive |
| exp19_typist: writing into a real sandbox .txt | 3/3 reach length-4 words; ~1000 words per lifetime | — | Pass; only the dial physics was learned, spelling loop hand-written |
| exp20_reader: one book streamed one character per tick | surprise 3.9 → 1.75 nats; seen-word completion ~30–40%; NEVER-seen words 0/N throughout | offline control (random access): held-out 2.23–2.36 (best 1.80) vs online 2.49–2.51 | Pass for emergence; online costs; batch 8 gives 3.6–3.7 vs batch 64 gives 2.5 |
| exp21_scratchpad: column addition on a visible pad | pad rule (2.2k params) 1.00 at every length; all-digits net 1.00 / 0.6–0.7 / 0.3 at 1 / 2 / 3 digits, 0.00 beyond | window-3 with no state 1.00 → 0.50–0.53 (carry chains) | PASS 3/3 |
| exp22_bench: writing programs for a subleq machine by imagination | v2 mastery 0.95–0.98; ~3,400 world-verified programs per life at 96% first-attempt success; coder 1.00 in 17/18 cells | v1 mastery 0.15–0.25; per-slot-marginal coder 0.0–0.12 | PASS 3/3 (v2) |
| exp23_pointer: can Rule G earn the pointer view? | pointer held-out CE 0.003–0.004 vs none 0.036–0.044 (ratio ≈ 0.1); GROW 3/3 | acts control rejected 2/3 (seed 2 ratio 0.73); blind ablation mastery 0.77 vs 0.96 | Earned |
| exp24_calendar: which past should sleep replay? | mean exam CE (3 seeds): prio 5.0 (200 slots) / 3.1 (800); uniform 5.6 / 3.9 | no replay = fifo 8.2; admit-only-surprising 5.8 | Pass: replay what you are forgetting |
| exp24b_overruled: does prio defend an obsolete rule? | prio relearn 702–1147 ticks, retention 4.85–5.46 | uniform relearn 810–919, retention 5.56–5.81; evidence-based stale mask 685–1079 (within noise) | Dose, not mask, is the current answer |
| exp25_locked: can a locked core plus a plastic head learn novel structure? | locked + head: C 2.8–6.1 nats, never recovered | plastic core 0.002–0.028, recovered in 1090–1550 ticks | Falsified: locked-core mode retired |
| exp26_stochastic: planning in a world that rolls dice (p = 0.6) | closed-loop sampled 0.38–0.48 (⅔–⅘ of oracle) in 3/3 seeds | oracle 0.59; random 0.00–0.08; v1 independent nibble heads gave miscalibrated 0.06–0.70 | Pass: autoregressive nibbles and a shared die needed |
| exp27_keypad: can curiosity plan its way to ignorance? | planning opens the door 202–215× (16.8–17.9 per 1k); room mastered at 2739–2935 in 3/3 seeds; open-tick error 0.009–0.029 | bandit-only opens 7–13× per 12k; room mastered 1/3 (at 11.8k) | PASS 3/3 |
| exp28_readergrowth: would the reader's probes ask for growth? | small diary: ctx128 ratio 0.73–0.82; big diary: ctx128 0.99–1.00, deep 1.03–1.11, wide 1.21–1.22 | small diary: deep 1.66–1.68, wide 2.29–2.46 (worse) | Mixed: capacity never wins a probe contest at equal steps |
| exp29_repl: does the reply loop close for text? | `?` learned 0.97–1.00 by every arm within 2k ticks; exact replies curious k=1/2/3/4: 1.00 / 0.93–1.00 / 0.40–0.49 / 0.00–0.12; bandit: 0.76–1.00 / 0.00–0.24 / 0.00 / 0.00 | bandit random typist; oracle k=3 0.64–0.84 | Loop closes; curious asks 3–4× more valid questions |
| exp29 goal exam (≤ 3-operand lines) | curious 0.39–0.54 | bandit 0.12–0.25; chance 0.05–0.09; bar 0.8 | FAIL the bar |
| exp30_thinkbytyping: the REPL as a pad, run in imagination | k=3: direct 0.37–0.52 → chained 0.80–1.00; k=4 (never lived): direct 0.01–0.18 → chained 0.65–1.00 | direct answers | PASS 3/3 (decomposition hand-written) |
| exp31_words: working-memory horizon with hidden colours | oracle16 0.66–0.72 within 14 keys, 0.40–0.50 beyond; oracle64 0.59–0.62 within 14, 0.44–0.54 at 15–62 | chance 0.33; curious arms at chance (born16 0.26–0.37, born64 0.26–0.33, grow16 0.33–0.42); 0 growths 3/3 | Horizon equals H; 4× H reads no farther; curious arms FAIL |
| exp32_habits: proposals from own habits, scored by entropy or bend | within-H what-accuracy 0.35–0.49 in every arm | bar 0.6 (chance 0.33) | FAIL |
| exp33_infogain: KL score (expected information gain, cheap form) | within-H 0.34–0.42 (kl), 0.39–0.54 (kl_eps); 88–180 pairs per life; 7,200–8,100 junk lines | typist: ~4000 pairs | FAIL |
| exp34_schema: proposals with the shape of an experiment | pairs 477–862 per life (bar met 6/6); within-H 0.65 / 0.37 / 0.44 (schema_kl), 0.52 / 0.47 / 0.65 (schema_eps) | accuracy bar 0.6 | Pairs PASS; accuracy FAIL 3/3 |
| exp35_templates: bandit learns which way of asking pays | schema template share 0.30–0.37 in the last 20k ticks; pairs 204–330; within-H 0.41–0.43 | given schema: about 2× the pairs | FAIL: the bend cannot reward a way of asking that has not yet paid |
| exp36_lookup L1: a key that carries its neighbours (one-hop lookup) | ngram64: 0.91–0.95 (d ≤ 14), 0.87–0.92 (15–30), 0.81–0.92 (31–62) | acts64: 0.50–0.59, 0.42–0.48, 0.41–0.43 | L1 PASS 3/3 |
| exp36_lookup L2 (ngram16 within 14 keys) | 0.94 / 0.75 / 0.84 | — | 2/3 seeds at the bar (per README) |
| exp36_lookup grow64 (born blind; can the contest earn a view?) | no view grown in any seed (`views: []`, `growth_events: []`) | — | Not earned (README still reads "[pending]"; result files show no growth) |
| exp37_ssm S1–S3: a gated recurrent state over the keys | ssm64 0.95–0.98 across its 64-key horizon; acts64 0.41–0.43 at 31–62; ssm256 0.98–0.99 out to 254 keys | acts64 at equal H | PASS 3/3 |
| exp37_ssm grow256 (born blind, H = 256) | no history organ grown in any seed (`views: []`, `growth_events: []`) | — | Bar G not met (README still reads "[pending]"; result files show no growth) |
| babybrain regression (`demo3.py`, v3c, ARCHITECTURE.md §6) | 19/21 core checks pass; world-latch grown 2/3 seeds (t = 7500, ratio 0.25 and 0.71), never in seed 2 (ratio 0.80–0.86 ×3) | — | Partial: failure is a Rule R (summary view) question |

## What it showed
- The laws that held in every byte world tried: descend rather than gate writes (exp1, exp1c, exp1d, exp16); replay the moments you are forgetting, with a dose that is large enough (exp2, exp24); grow a view when a probe contest says it helps, not when the error curve bends (exp3, exp4, exp23); plan by imagined rollouts (exp5, exp12, exp26, exp27, exp30). The lab's "falsified" ledger (THEORY §9) lists 16 dead ideas, and they stay listed.
- Memory length is not the fix. A raw list of the last 64 keys read no farther than 16 keys (exp31, 0.59–0.62 within 14 keys vs 0.66–0.72 for 16 keys). A carried state read 254 keys at 0.98–0.99 (exp37), and a key that carries its neighbours reached 0.81–0.95 out to 62 keys (exp36). These results are measured within the byte worlds tested.
- Curiosity did not learn to design experiments. The learning-progress drive and three information-style scores (entropy, bend, KL) all failed to produce intervention-then-query experiments in a language (exp31–33). With an experiment-shaped proposal language the pair count rose 5–8× (exp34), but the accuracy bar still failed in 3/3 seeds, and a bandit could not learn to choose that shape (exp35).
- Locked cores cannot learn novel structure (exp25). Language reply loops close, grammar first, and curiosity asks more questions, but the goal exam at 0.8 failed (exp29). Ten-digit operands stayed beyond one life in exp29 (k=2 at 0.06–0.18 after 40k ticks).
- Growth is the least trusted piece. Probe-triggered growth worked in controlled cases (exp3, exp23), but the growth arms born blind in exp36 and exp37 grew nothing across 3 seeds each, and the demo3 world-latch grew in 2 of 3 seeds.

## Limitations
- Scale: the organism is ~137–140k parameters. Experiments at 10× (730K, exp1d) and ~9M (exp16) are single probes of the gate question, not full runs of the organism.
- Worlds: most worlds are ≤ 64 bytes and written by the author for the lab. Several proofs rest on them (THEORY §18.14, PHASE1_SUMMARY §4).
- Seeds: three seeds per arm. Several results have a wide spread (for example exp10 2/3 and exp34 "two lives of six").
- Missing baselines: the growth verdict has no held-out floor (listed as an open next step in the handoff). exp18's own-effect test was inconclusive, so the actuator effect was never shown.
- Hand-written scaffolding: exp19 and exp21 spelling and carry loops, and exp30's decomposition, are written by hand, so those results show what the learner does with the scaffold, not that it invents one.
- Revised claims: the exp1 "recovery ~200" claim came from a pre-bugfix run and was retracted; the post-bugfix rerun is the one reported.
- The organism is a state predictor, not a language model. Its core must be rewritten for sequences before the language results transfer (PHASE1_SUMMARY §5).
- Documentation mismatch: `experiments/README.md` marks exp36 grow64 and exp37 grow256 as "[pending]". Their result files show no growth in any seed. This log uses the result files.

## Key files
- `experiments/babybrain/brain3.py`: the organism (BabyBrain3): core, views, descent, naps, bandit, growth probes, planner, save/load.
- `experiments/babybrain/demo3.py`: the regression (nursery, latch, buttons, world-latch × 3 seeds).
- `experiments/babybrain/world.py`: the 16-region nursery world.
- `experiments/README.md`: the lab log, one row per experiment (exp0–exp37) and the handoff paragraph.
- `THEORY.md`: the theory (v5): seven rules (§0), falsification ledger (§9), sections 10–27 per experiment era, critique (§18), goal restated (§27).
- `ARCHITECTURE.md`: the organism spec (v3); §5 two-timescale diary; §6 regression table.
- `PHASE1_SUMMARY.md`: the summary written when phase 1 closed (laws, falsified list, honest position).
- `experiments/expNN_*/run.py`: each experiment's driver; its docstring holds the design and the pre-registered bar. Results are in `result_<seed>.json` and logs in `log_<seed>.txt`.
- `experiments/exp36_lookup/run.py`, `experiments/exp37_ssm/run.py`: the memory-organ experiments (exp37 reuses exp36's driver).

## Sources
- `experiments/README.md`: all per-experiment numbers in the Results table, unless another source is named.
- `PHASE1_SUMMARY.md`: architecture summary, laws table, falsified list, honest position, hardware line for phase 1 dates.
- `THEORY.md`: seven rules (§0), falsification ledger (§9), ancestry (§8), named sections cited in the table.
- `ARCHITECTURE.md`: §5 (two-timescale diary), §6 (regression table, RTX 4080 run time).
- `experiments/exp36_lookup/result_0.json`, `result_1.json`, `result_2.json`: grow64 arm (no views grown).
- `experiments/exp37_ssm/result_0.json`, `result_1.json`, `result_2.json`: grow256 arm (no views grown); parameter count (136,928).
- `experiments/exp27_keypad/result_0.json`: spot-checked bandit and planning arm values against the README row.
- `experiments/exp37_ssm/run.py`: pre-registered bar (S1–S3, G).
