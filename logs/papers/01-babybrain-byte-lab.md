# Falsifying the Parts of a Small Self-Learning Organism: A Phase-1 Byte-World Study

Eric Lam (independent researcher, Toronto)

## Abstract
We report phase 1 of a study of a small learner that adapts from its own experience with one prediction loss and no task labels. The organism (`BabyBrain3`, about 137k parameters in the reported runs) was tested in 38 falsifiable experiments (exp0–exp37) in byte worlds of at most 64 bytes, plus a few longer streams. Each experiment states a pass/fail bar before its run, uses three seeds, and compares against a control. Several design choices were falsified: write gating of memory, curiosity that scores its own replies, and growth from a blind start. Others held: replaying the moments being forgotten, growing a view by probe contest, and planning by imagined rollouts. Memory length alone did not extend recall, while a compressed recurrent state read 254 keys at 0.98–0.99. We also report several misses, including curiosity-driven experiment design, which missed its accuracy bar in 3 of 3 seeds. The results are measured in worlds we wrote, at small scale.

## 1 Introduction
The question is whether a model that stays small enough for an ordinary device can keep learning from its own experience. The study took one design constraint from that goal: all adaptation is gradient descent on one loss, with a schedule chosen by the learner. Beyond that constraint, each design choice was treated as a hypothesis to test. The goal is not a benchmark score. It is a set of rules that either survived a controlled test or were removed.

Phase 1 covered 2026-08-13 to 2026-08-16. This paper reports the research half of phase 1. The project's theory document keeps a falsification ledger of 16 dead ideas, and those are not retried.

## 2 Method
**Organism.** `BabyBrain3` (`experiments/babybrain/brain3.py`) is a two-block full-attention transformer over state tokens. Each observation byte becomes one token holding its symbol, bits, scalar value, position, and action. Heads are autoregressive over nibbles (high nibble first, then low nibble given high). Temporal views (an action history `acts`, a per-region observation summary `obs`, a neighbour-lookup `ngram`, and a gated recurrent state `ssm`) are added only when a probe contest favours them.

**Learning.** Online Adam steps run on fresh and recent moments. Every 1000 ticks a "nap" replays a lifelong reservoir, drawn in proportion to current error. The two timescales are the design's answer to the observation that a diary of recent moments cannot distinguish forgetting from change.

**Worlds.** Byte worlds of at most 64 bytes include a 16-region nursery with a flipping needle, a two-button world with dice, a lamp-and-latch world, a subleq machine, a keypad whose four-press code opens a room, a four-key number REPL, and a world with hidden colours that are changed and queried by words. Longer streams include one book (143k characters) read one character per tick, and a live host-telemetry stream.

**Controls.** Each experiment names its control: random actions, no gate, no replay, first-in-first-out replay, offline random access, a shuffled buffer, a blind organism without the view under test, flat search, or an oracle.

**Seeds and bars.** Three seeds per arm. Bars are written in each experiment's driver docstring before the run; a missed bar is reported as a miss.

## 3 Experiments and Results

### 3.1 Learning rules

| Experiment | Result | Control | Verdict |
|---|---|---|---|
| exp1c, write gate under a rule flip | gated head recall-only; never re-acquires the old rule | no gate: steady 0.028–0.035, re-acquisition 420 → 160 ticks | Gate falsified |
| exp1d, same switch at ~10× core (730K) | no-gate steady 0.031–0.034; gate 0.24–0.26 at 8% writes | no-gate | No-gate still wins |
| exp16, frozen switch at ~9M (GPU) | gate 0.23 | no-gate 0.037 | No-gate still wins |
| exp2, forgetting at 60K params | 8.7 nats → 1.04 with a 500-item reservoir and naps | no replay | Pass (8×) |
| exp24, which past to replay (exam CE, 3 seeds) | prio 5.0 (200 slots) / 3.1 (800) | uniform 5.6 / 3.9; no replay 8.2 | Pass |
| exp24b, replay of an overruled lesson | prio relearn 702–1147 ticks | uniform 810–919; evidence-based mask 685–1079 | Mask within noise |
| exp25, locked core with plastic head | C 2.8–6.1 nats, never recovered | plastic core recovered in 1090–1550 ticks | Falsified |

### 3.2 Growth and memory

| Experiment | Result | Control | Verdict |
|---|---|---|---|
| exp3, probe-triggered growth | 3/3 correct growths, 0 premature, 0 noise growths | curve-read triggers falsified | Pass |
| exp4, memory grown for a 16-tick latch | flip CE 1.39 (blind floor) → 0.003–0.021 | GRU stuck at 1.70 | Pass |
| exp23, pointer view earned by contest | pointer CE 0.003–0.004 vs none 0.036–0.044 | acts control rejected 2/3 | Pass |
| exp31, working-memory horizon, oracle diary | oracle16 0.66–0.72 within 14 keys, 0.40–0.50 beyond | oracle64 0.59–0.62 within 14, 0.44–0.54 at 15–62 | Horizon equals H |
| exp36, neighbour-lookup key (L1) | ngram64 0.81–0.95 across 14/30/62 bins | acts64 0.41–0.59 across the same bins | Pass 3/3 |
| exp37, gated recurrent state | ssm64 0.95–0.98 at 64; ssm256 0.98–0.99 out to 254 keys | acts64 0.41–0.43 at 31–62 | Pass 3/3 |
| exp36 / exp37, blind growth arms | no view grown in any seed | — | Not met |
| demo3 regression (v3c) | 19/21 core checks; world-latch grown in 2/3 seeds | — | Partial |

### 3.3 Planning, language, and curiosity

| Experiment | Result | Control | Verdict |
|---|---|---|---|
| exp5, planning by imagined rollouts | 98–100% | random 16–24%; single button 0% | Pass |
| exp26, planning under dice (p = 0.6) | closed-loop sampled 0.38–0.48 | oracle 0.59; random 0.00–0.08 | Pass |
| exp27, curiosity plans to open a door | 202–215 openings (16.8–17.9 per 1k); room mastered 3/3 | bandit 7–13 per 12k; mastered 1/3 | Pass |
| exp30, REPL as an imagined pad (k = 3 / k = 4) | chained 0.80–1.00 / 0.65–1.00 | direct 0.37–0.52 / 0.01–0.18 | Pass (decomposition hand-written) |
| exp29, reply loop for text | `?` learned 0.97–1.00 by every arm; curious k = 3 exact 0.40–0.49 | bandit k = 3 0.00; oracle k = 3 0.64–0.84 | Loop closes; goal exam 0.39–0.54 vs bar 0.8 (miss) |
| exp31, curiosity with word-keys | curious arms 0.26–0.42 | chance 0.33 | Miss |
| exp32–33, three scores on the reply | within-H 0.34–0.54 | bar 0.6 | Miss |
| exp34, experiment-shaped proposals | 477–862 pairs per life (bar met 6/6); accuracy 0.37–0.65 | accuracy bar 0.6 | Miss on accuracy 3/3 |
| exp35, bandit over ways of asking | schema template share 0.30–0.37 | — | Miss |

## 4 Discussion
Three findings are the most useful for a next phase.

First, memory is compression. A raw history of 64 keys did not read farther than 16 keys (exp31). A neighbour-carrying key reached 62 keys with 0.81–0.92 (exp36), and a gated recurrent state reached 254 keys with 0.98–0.99 (exp37). The state result is the strongest in the study, measured in a byte world with a known oracle.

Second, the curiosity signal is the weakest part. Entropy, learning-progress "bend", and a KL score all produced fewer useful intervention-query pairs than a typist. In a language, novel nonsense is as uncertain as an unknowable colour, so the drive chases noise. Only a proposal language with the shape of an experiment raised the pair count (5–8×, exp34), and a bandit over proposal templates could not learn which shape pays (exp35).

Third, growth has no reliable trigger yet. Probes can grow a view when the view is the right one (exp3, exp23). But a blind organism given the chance to grow a history view grew nothing in 3/3 seeds (exp36, exp37), and the demo3 world-latch grew in 2/3 seeds. The verdict cannot yet separate "young core" from "blind core". The growth verdict has no held-out floor in the current design.

These results support a narrow claim: several local rules hold in small byte worlds, and several plausible extensions do not. They do not establish that the same rules transfer to language models at scale.

## 5 Limitations
- **Scale.** About 137–140k parameters in the main organism. The 730K and ~9M runs are single probes of the gate question.
- **Worlds.** Most worlds are at most 64 bytes and were written for the study. Several proofs rest on them.
- **Seeds.** Three seeds per arm. Some results are wide (exp10 2/3; exp34 accuracy 0.37–0.65).
- **Scaffolding.** Spelling loops (exp19, exp21) and the REPL decomposition (exp30) are hand-written. Those results show what the learner does with the scaffold.
- **Model type.** The organism is a state predictor over bytes, not a language model. Its core would need rewriting for sequences before language results transfer.
- **Documentation.** The lab log marks two grow arms as pending. Their result files show no growth. This paper uses the result files.

## 6 Conclusion
A small learner with one prediction loss kept several of its rules and lost several others. Replay of what is being forgotten, probe-based view growth, and imagined planning survived controlled tests. Write gating, blind-start growth, and curiosity-driven experiment design did not. The most durable positive result is that a compressed recurrent state reads far longer histories than a raw list of the same horizon, in the byte worlds tested.

## References
The project documents name the following works. They give no full citations, years, or venues, so none are supplied here.
- Oudeyer, learning-progress construction (named in THEORY.md §8 as the origin of Rule A).
- ALP-GMM, the absolute-value form of learning progress (THEORY.md §8).
- SIESTA and DER, replay methods that Rule S is compared with (THEORY.md §8).
- Titans, a fast-weight memory with surprise-gated writes (THEORY.md §8; the study's write-gate result is set against it).
- The fast-weight literature more generally (THEORY.md §8).
- Adam, the optimizer used for online descent (ARCHITECTURE.md, phase-1 summary).
- GRU, the recurrent baseline in exp1 and exp4 (experiments/README.md).
