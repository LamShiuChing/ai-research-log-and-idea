# Self-improvement loops: when does a model improve itself vs drift?
**Dates:** Not stated in the docs (source files last modified 2026-09-05) · **Status:** concluded (toy scale; no later work in the source folder) · **Hardware/scale:** Not stated. Toy tasks: 8-bit strings, targets 0 to 99, number words 0 to 99, six-letter word streams. The neural runs (`selfloop3.py` to `selfloop5.py`) use a 128-unit GRU and run on CUDA if available, otherwise CPU.

## Question
When a model generates candidates, checks them, and updates itself on what passed, does it get better at the task or drift toward its own mistakes? The working hypothesis in `THEORY.md` is that the loop stays sound only when correctness comes from an external check, the archive of verified items only grows, and exploration is built into the proposals.

## Approach
- **Models:**
  - Toy: a categorical distribution over all 256 eight-bit strings (`selfloop.py`).
  - Follow-ups: a 256-parameter table, an 8-parameter independent-bit model, and a 72-parameter counting model (`selfloop2.py`).
  - Neural: a character-level GRU with 32-dimensional embeddings and 128 hidden units (`selfloop3.py` to `selfloop5.py`).
  - No neural network: a symbol-inventing context model (`selfloop6.py`), and a program learner with particle-based hypotheses and an archive re-described by Re-Pair-style merges (`selfloop7.py`, `selfloop8.py`).
- **Tasks:**
  - Emit an 8-bit string with exactly four ones (70 of 256 strings are correct).
  - Grow a learned verifier from a small oracle budget.
  - Propose own tasks: 72 tasks, 28 of them impossible.
  - Arithmetic: write an expression over digits 1 to 9 and `+ - *` (up to four operands) that equals a target from 0 to 99. Python is the teacher; 20 targets are held out.
  - English: write a number from 0 to 99 in words. A dictionary is the teacher; 40 of the 80 training numbers are taught by hand; 20 are held out.
  - Thinking: evaluate an expression such as `7*8-3+2` (answer 55), with up to 8 private pad symbols before the answer.
  - Word stream over six letters, with hidden words from lexicon A, then B, then A again.
  - A stack-machine program language with four hidden macros (depth 1 to 3), and a two-level library (macros and phrases).
- **Method:** a loop of sample, verify, update. `selfloop.py` compares 13 update rules on the toy task: `rft`, `self_judge`, `self_distill`, `reinforce`, `reinforce_entropy`, `trust_region`, `reward_model`, `evolve`, `anchor`, `smooth`, `greedy`, `Unified` (model 12, the proposed rule), and `unified_proxy` (the same rule with a learned verifier). The proposed rule keeps a monotone archive of verified items, proposes from mutations and uniform samples, answers from the uniform distribution over the archive, and stops when the discovery rate plateaus.
- **Controls and baselines:** each test names its baseline in the Results table. The main comparisons are the verified loop against self-judging, no-archive, no-exploration, learned-verifier, and uniform-token variants.
- **Seeds:** one fixed seed per script, with no aggregation over seeds in the code: `selfloop.py` 0; `selfloop2.py` 1; `selfloop3.py` to `selfloop5.py` 0; `selfloop6.py` 0; `selfloop7.py` 1; `selfloop8.py` 2.

## Results
| Test | Result | Baseline / control | Verdict |
|---|---|---|---|
| Toy 8-bit task, `Unified` (model 12) | Stops at round 5 to 7 "on every seed" (per `THEORY.md`); 100% correct at the stop | Other rules: no per-rule numbers in the docs; `THEORY.md` describes self-reference, drift, Goodhart and degenerate-optimum failures qualitatively | Supported (toy). The "every seed" claim is not backed by the code, which fixes seed 0 |
| Toy, `unified_proxy` (learned verifier) | Stops at round 5, fully confident, 25% correct | `Unified` with the true verifier | Negative: the loop cannot exceed its verifier |
| Grow verifier, 2 oracle labels per round | Learned proxy went from 72% accurate to exact within 190 labels; filled the archive twice as fast as random auditing | Random auditing; a frozen proxy (never improved) | Supported (toy) |
| Shrink learner, 256-parameter table | Memorises the archive and never leaves it | 72-parameter counting model | Negative |
| Shrink learner, 8-parameter independent-bit model | Proposes at chance indefinitely (wrong inductive bias) | 72-parameter counting model | Negative |
| Shrink learner, 72-parameter counting model | About 50 examples give 90% correct proposals; archive filled in 3 rounds | 256-parameter table; 8-parameter model | Supported (toy). "Bias beats size" |
| Propose own tasks, reward = solver success | Curriculum narrows to tasks already solved | Uniform curriculum | Negative (qualitative; no number in docs) |
| Propose own tasks, reward = failure | 77% of the budget spent on impossible tasks | Uniform curriculum | Negative |
| Propose own tasks, reward = learnability s(1-s) | Matches uniform curriculum on competence; cannot tell "hard" from "impossible" | Uniform curriculum | Null (no advantage over uniform) |
| Propose own tasks, learnability plus executor validity check (Absolute Zero recipe) | Zeroes the waste on impossible tasks (per `THEORY.md`) | Learnability alone | Supported (no number in docs) |
| Arithmetic, verified, no exploration | Train targets 0.78; held-out, student alone 0.01; held-out, student plus teacher pass@100 0.15; notebook 441 | Verified plus exploration | Negative: memorises the seed set |
| Arithmetic, verified plus exploration | Train targets 0.31; held-out, student alone 0.04; held-out, student plus teacher pass@100 0.85; notebook 11,500 | Verified without exploration | Supported (teacher plus student). The student alone does not learn the arithmetic |
| Arithmetic, self-judge | Train targets 0.18; held-out, student alone 0.01; held-out, student plus teacher pass@100 0.30; notebook 451 | Verified plus exploration | Negative |
| Arithmetic, no archive | Train targets 0.45; held-out, student alone 0.05; held-out, student plus teacher pass@100 0.10; notebook 425 | Verified plus exploration | Negative |
| English, verified | Notebook 72 / 80 after 30 rounds; untaught numbers, student alone 0.80; held-out, student alone 0.60; held-out pass@20 0.80. Verified student composed 32 of 40 untaught numbers correctly | Self-judge | Supported. Errors fall where English is irregular: writes 15 as "fifty-one"; never found 10, 11, 13, 14, 17, 18, 19 |
| English, self-judge | Notebook 44 / 80; untaught 0.10; held-out alone 0.15; held-out pass@20 0.30; "thinks" rose 0.50 to 0.96 | Verified | Negative: confident while stuck |
| Thinking, free pad, monotone notebook | Held-out 0.05; own / shuffled / random pads 0.06 / 0.06 / 0.04; 2689 distinct pads; dictionary reads 0.01 of held-out answers | Shuffled and random pads | Negative: the pad carries no content |
| Thinking, free pad, latest per task | Held-out 0.05; own / shuffled / random 0.05 / 0.05 / 0.04; 1957 distinct pads; dictionary 0.00 | Shuffled and random pads | Negative |
| Thinking, free pad sampled at T=0.5 | Held-out 0.05; own / shuffled / random 0.04 / 0.04 / 0.05; 1811 distinct pads; dictionary 0.02 | Shuffled and random pads | Negative |
| Thinking, reader sees only the pad (10^8 messages) | Held-out 0.03; own / shuffled / random 0.02 / 0.03 / 0.03; 1505 distinct pads; dictionary 0.02 | Shuffled and random pads | Negative |
| Thinking, 100 messages (2 symbols of 10) | Held-out 0.03; own / shuffled / random 0.03 / 0.02 / 0.03; 99 distinct pads; dictionary agrees with reader 0.38 (0.37 after 100 rounds) | Shuffled and random pads | Negative: no shared code. Note: `THEORY.md` says writer and reader "never agreed", which does not match the 0.38 figure in the same table; unresolved |
| Thinking, no pad | Held-out 0.13 | Free pad (8 self-written symbols) 0.05 | Supported as a comparison: free pads hurt |
| Thinking, fixed pad "aaaaaaaa" | Held-out 0.23 | No pad 0.13 | Supported: constant filler helps |
| Word stream, phase A | Bits per letter 2.41 (first 300 tokens) to 1.98 (last 300); 48 symbols invented | n/a | Supported |
| Word stream, phase B | Bits per letter 3.79 to 1.69; 112 symbols invented | n/a | Supported |
| Word stream, A again | Bits per letter 2.15 to 1.96; 133 symbols invented; first window of A costs 2.15 bits per letter on return vs 3.79 when B was new | Phase B cost | Supported (retention) |
| Word stream, symbols vs hidden words | 110 of 133 invented symbols are whole hidden words or runs of them; all 16 hidden words became symbols | n/a | Supported |
| Word stream, sampled text | 1.00 of sampled letters covered by hidden words; 33% of 12-letter windows never occurred in the input | Random letters: 0.26 | Supported (valid and new) |
| Recursion, learning-progress policy | World bits per letter 1.25 (turn 200) and 0.92 (turn 600); share of turns on noise action 0.08 and 0.39; stopped at turn 342 | Raw surprise; unplugged policy | Supported |
| Recursion, raw-surprise policy | World bits per letter 2.12 and 1.81; noise share 0.47 and 0.92; never stopped | Learning progress | Negative: noisy-TV failure |
| Recursion, unplugged at turn 200 | World bits per letter 1.28 and 1.39; never stopped; 27 more symbols from its own babble | Connected policies | Negative: no new knowledge of the world |
| Program language, depth 1 / 2 / 3 solved | Full (language plus invented words) 0.90 / 0.80 / 0.35; samples per episode 2041 / 4080 / 4876; 18 words | No-words 0.60 / 0.30 / 0.05; random tokens 0.85 / 0.20 / 0.00 | Supported at depth 2 and 3. At depth 1 the random baseline is close (0.85 vs 0.90, 20 episodes per depth) |
| Library, macro block | Full 1.00 (1183 samples per episode; 26 words) | No-words 1.00 (1534); random 0.90 (4649) | Supported (cheaper, not better) |
| Library, phrase block | Full 1.00 (1235 samples) | No-words 1.00 (1488); random 0.45 (4742) | Supported |
| Library, two phrases | Full 1.00 (2750 samples) | No-words 0.25 (4953); random 0.10 (4988) | Supported: composition needs the words |
| Library, macro again | Full 1.00 (449 samples; retention vs 1183 first time) | No-words 1.00 (650); random 0.95 (4486) | Supported (retention) |

## What it showed
- With an archive of verified items and an external check, the toy loop converged to fully correct output and stopped on a discovery signal, not on belief. Rules that trained on their own outputs or on a learned judge did not.
- The loop is only as good as its verifier. A learned judge made the loop stop fully confident at 25% correct. The same loop with a learned verifier that is refined by audits reached an exact verifier within 190 labels.
- Exploration is necessary. Without it the arithmetic student memorised the seed pairs and reported nothing new. With it, the notebook grew to 11,500 verified answers, and the student plus teacher solved 0.85 of held-out targets at pass@100, against 0.15 without exploration. The GRU student alone stayed between 0.01 and 0.05 on held-out targets across variants, so the loop did not give it arithmetic. (`THEORY.md` says the precision "stayed near 4% in every variant", which the table does not show.)
- Free-form "thinking" carried no content. Shuffled and random pads scored the same as the model's own. Constant filler helped (0.23 vs 0.13), which matches the idea that extra compute can help without a message.
- In the non-neural learners, invented symbols matched the hidden structure, and composed programs needed the invented words. The random-token baseline was competitive at depth 1, so the advantage shows at depth 2 and 3.

## Limitations
- Toy scale throughout. The arithmetic, English and program tasks are small; the verifiers are exact; nothing is tested on a real language model or a real checker such as tests or proofs.
- One fixed seed per script, with no error bars. `THEORY.md` says the toy result holds "on every seed", but `selfloop.py` runs only seed 0.
- No output logs are in the source folder. Every number here is copied from `THEORY.md`, which summarises printed runs. The scripts were not re-run for this log, so the printed values were not checked against the tables.
- `THEORY.md` says the arithmetic student was tested "at two model sizes". The code shows only the 128-unit configuration.
- Several claims are qualitative in the docs: the narrowing of the task curriculum under a success reward, and the Absolute Zero validity check. These are listed as such in the table.
- The "theorem" in `THEORY.md` (almost-sure convergence in coupon-collector time when epsilon is greater than 0) is stated, not proved in the files.
- `THEORY.md` says "eleven update rules"; `selfloop.py` lists thirteen models, with the rule numbering running 1 to 12.
- Primitives, merge thresholds, exploration shares and "shorter wins" are hand-fixed in every file.

## Key files
- `THEORY.md`: the write-up. Theory of the loop, the four failure modes, the proposed update rule, and summaries of every follow-up.
- `selfloop.py`: the toy 8-bit task and 13 update rules compared head to head, with a stop criterion based on the model's own belief.
- `selfloop2.py`: three follow-ups. Growing the verifier from an oracle budget, shrinking the learner (256-parameter table, 8-parameter model, 72-parameter counting model), and a model that proposes its own tasks under five reward rules.
- `selfloop3.py`: a character GRU trained on a notebook of verified arithmetic expressions, with Python as teacher. Compares verified, verified plus exploration, self-judge and no-archive variants.
- `selfloop4.py`: the same GRU on English number words, with a dictionary as teacher and 40 of 80 training numbers taught by hand.
- `selfloop5.py`: the GRU given private pad symbols before answering. Tests whether the pad carries content, whether it can be read as a code, and compares free, constant and two-symbol pads.
- `selfloop6.py`: a learner with no neural network. It invents symbols by merging pairs under a description-length criterion, and learns from a stream of hidden words and from its own actions in a small world.
- `selfloop7.py`: the program learner. A stack-machine language with hidden macros, particle-based hypotheses, Occam deletion, and invented words, compared with no-words and random-token modes.
- `selfloop8.py`: the program learner with an archive re-described by merges, a two-level hidden library (macros and phrases), and block-by-block retention tests.

## Sources
- `THEORY.md`: all numbers in the Results table, and the claims about the "every seed" result, the 77% figure, the 190 labels, and the 3-round and 50-example results.
- `selfloop.py`: the setup, the 70 of 256 correct-string definition, the list of 13 models, and the seed.
- `selfloop2.py`: the 72-task and 28-impossible setup (printed headers), the 256-parameter, 8-parameter and 72-parameter proposers, and the seed.
- `selfloop3.py`, `selfloop4.py`, `selfloop5.py`: model size (hidden 128), the held-out split, and the teacher definitions.
- `selfloop6.py`, `selfloop7.py`, `selfloop8.py`: the hidden-structure definitions, the tested modes, and the seeds.
