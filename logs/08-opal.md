# OPAL: DreamCoder-style program induction with a learned library and LLM proposers
**Dates:** Sep 6–8, 2026 (docs dated 2026-09-06 to 2026-09-08) · **Status:** ongoing (last dated entry 2026-09-08; open items listed at the end of RESULTS.md) · **Hardware/scale:** Windows 11, Intel Core i9-14900KF, NVIDIA RTX 4080 SUPER (local Ollama models); grid and dial worlds of 12–14 hidden rules per curriculum, 20–50 s search per task, four seeds per paired condition

## Question
Does a growing library of executable abstractions, kept with an otherwise identical agent that is denied abstraction, reduce later search cost and raise the number of rules learned? Do the gains transfer to other domains, and do LLM proposers add anything to the same search?

## Approach
- **Models and proposers:** No neural network is trained in the core learner. Proposers are a local Qwen 2.5 14B (dial pieces, 2026-09-07), retrieval of pieces from past solutions (no model), a small dream-trained guide (neutral), and local Qwen 2.5 7B / Gemma 3 4B for coding-task proposals.
- **Worlds:** a grid world with a single agent (12 hidden rules, `RESULTS.md` "Seed 0") and a dial-panel domain of 10–14 rules (`opal/machines.py`). Rules are hidden programs; the agent sees observations and chooses actions.
- **Method:** the library agent refits grammar weights from usage and runs anti-unification abstraction between tasks ("sleep"). The control is the same agent with sleep removed, so it keeps solutions and weights but never abstracts. The split driver (`opal/split.py`) recursively splits a task before search.
- **Controls and baselines:** paired library vs no-abstraction control on identical rules, layouts and budgets; "cache" and "reset" controls on seed 0 only; a frozen-solver transfer run on rule orders from seeds 4–7; cross-domain transfer (one agent through both worlds).
- **Tasks and metrics:** rules learned (interactions to 95% exact next-state prediction on a stopping set), accuracy on an independent trajectory, and lifetime search seconds.
- **Coding (H language, integer/list functions):** the LLM proposes programs in a small typed language, checked against public examples; independent tests are held out. Arms: LLM-only, LLM + fallback search, and LLM + search + library (single run, six functions). A generated-rule benchmark (`CODING-LEARNING-PROTOCOL.md`) uses three backends (Gemma 3 4B, Qwen 2.5 7B, no LLM) and three memory controls (fresh, experience, abstraction) over seeds 17, 29, 43.
- **Seeds:** four per paired grid/dial condition; the unseen-order transfer used seeds 4–7.

## Results

| Test | Result | Baseline / control | Verdict |
|---|---|---|---|
| Grid, 12 rules, enumeration-only search, seeds 0–3: mean rules learned (mean lifetime search s) | library 4.8 (451 s) | control 2.5 (573 s) | Library ahead; effect bimodal (see Limitations) |
| Grid, stronger search (D32), seeds 0–3: mean rules ≥95% on independent test (search s) | library 8.5 (200 s) | control 6.5 (489 s) | Library ahead |
| Grid, second round (D34–D38), seeds 0–3: rules ≥95% on test (search s) | library 9.0 (177 s) | control 8.0 (478 s) | Library ahead |
| Grid, one vs three layouts (D40), seeds 0–3: rules ≥95% on test | three-layout library 9.5 (197 s) | three-layout control 9.0 (459 s) | Library ahead, small margin |
| Grid, unseen rule orders, frozen solver (D45), seeds 4–7: rules ≥95% on test (search s) | library 9.50 (242 s) | control 8.25 (567 s) | Library ahead on fresh orders |
| Dial/machines domain (D45), seeds 0–3: rules learned (search s) | library 2.0 (931 s) | control 0.8 (1196 s) | Library ahead; both low |
| Dial, sixth round (D48–D50), seeds 0–3: rules learned / ≥95% on test (search s) | library 2.8 / 3.0 (1019 s) | control 2.0 / 2.0 (1295 s) | Library ahead; on-only turn rule 3 of 4 vs 0 of 4; colour-press 4 of 4 in both arms |
| Dial, split driver (D62), 20 s pinned, seeds 0–3: rules learned / ≥95% on test (search s) | library 4.2 / 4.5 (612 s) | control 3.8 / 4.2 (792 s) | Library ahead on every count |
| Stepping stones, first pass (D47): colour-press rules learned, seeds 0–3 | library 1 of 12 (3.0 learned, 464 s) | control 0 of 12 (2.8 learned, 611 s) | Null on the target colour rules |
| Stepping stones, second pass (D47): rules learned / ≥95% on test (search s) | library 3.2 / 3.8 (484 s) | control 3.5 / 3.2 (711 s) | Mixed; colour-press 1 of 12 in both arms |
| Retrieval of past pieces, dial (D53): rules learned / ≥95% on test (search s) | library + retrieval 3.0 / 3.0 (964 s); control + retrieval 2.0 / 2.0 (1277 s) | library 2.8 / 3.0 (1019 s); control 2.0 / 2.0 (1295 s) | Retrieval helps control on clamp (0.50 → 0.90) and decay (0.28 → 0.75); counts unchanged |
| Retrieval of past pieces, grid (D53): rules learned / ≥95% on test (search s) | control + retrieval 9.8 / 10.2 (382 s); library + retrieval 9.0 / 9.5 (201 s) | library 9.0 / 9.5 (206 s); control 8.2 / 8.8 (480 s) | Negative for the library: retrieval-only control learns more rules than the library |
| Reuse with challenge (D44), grid: rules learned / ≥95% on test (search s) | reuse + challenge 9.0 / 9.5 (199 s) | reuse off 9.0 / 9.5 (197 s) | Null |
| Evidence, 90 vs 45 interactions (D55), dial: library clamp / wrap test | 90 interactions: clamp 0.79, wrap 0.65 | 45 interactions: clamp 0.79, wrap 0.65 | Null for clamp and wrap |
| Min and max primitives (D59), dial, seeds 0 and 1: on-only turn | 0.88 on both seeds | baseline 1.00 on both seeds | Negative (clamp 0.99 and 0.87 unchanged) |
| Anomaly replay (D60), dial, seed 0: held-out accuracy in the probe | 0.27 after one replayed batch | 0.73 before the replayed batch | Negative as built (seed 1 identical to random); off by default |
| Cross-domain transfer (D63): dial rules learned / ≥95% / search s | dial after grid 4.00 / 4.25 / 611 s | dial alone 4.00 / 4.25 / 613 s | Zero effect |
| Cross-domain transfer (D63): grid after dial | grid after dial 9.00 / 9.50 / 146 s | grid alone 9.00 / 9.50 / 153 s | Zero effect |
| Agent loop (D65), dial, 20 s pinned: rules learned / ≥95% on test (search s) | nested loop 4.00 / 4.25 (1239 s) | harness 4.25 / 4.50 (612 s); flat loop 3.00 / 3.25 (1117 s) | Negative vs harness: fewer rules at about twice the search |
| Coding, six functions, single run (CODING-RESULTS): tasks passing all independent tests; LLM calls; generated tokens | LLM + search + library: 6/6; 7; 208 | LLM-only 6/6; 7; 236; LLM + search 6/6; 7; 236 | All arms solved all tasks; no advantage shown |
| Generated coding, acquisition (CODING-LEARNING-RESULTS), tasks of 18 passing all 64 final tests: Gemma 3 4B | fresh 2/18; experience 4/18; abstraction 4/18 | — | Memory helps; no library entries adopted in any run |
| Generated coding, acquisition: Qwen 2.5 7B | fresh 2/18; experience 4/18; abstraction 4/18 | — | Same pattern as Gemma |
| Generated coding, acquisition: no LLM (search only) | fresh 2/18; experience 4/18; abstraction 4/18 | — | Same numbers as with LLMs, so the gain is not from model proposals |
| Generated coding, transfer to new compositions (9 per backend/control) | 0/9 in every backend and control | — | Negative |
| Oracle-fed diagnostic (dev seed 0, privileged): new compositions passing all 64 tests | with consolidated memory 3/3 | cold search 0/3; same six correct programs without abstraction 0/3 | Positive, but correct training programs were supplied |
| H fluency probe: simple written requests with both LLMs | Qwen 2.5 7B 2/2 tasks; Gemma 3 4B 0/2 tasks | — | Gemma's acquisition result is confounded by H fluency |

## What it showed
- In the author's grid and dial curricula, the library arm was ahead of its paired control on search seconds in every comparison and on rules learned or rules ≥95% on test in most of them. The margin was about 0.5 to 2 rules in the grid and about 0.4 to 1.2 rules on the dial domain, and it moved with the search version and with the evidence budget.
- Most of the gain came from reuse of pieces, not from abstraction alone. Retrieving pieces of past solutions, with no abstraction at all, lifted the grid control to 9.8 rules learned and 10.2 at ≥95% on test, above the library arm's 9.0 and 9.5.
- Cross-domain transfer was zero in both directions. Each library entry named its own domain's verbs or attributes, so nothing was reusable.
- Several hard rules (clamp, wrap, decay, linked, and the panel fold) stayed unsolved. Doubling evidence, adding min and max primitives, and a goal-directed loop did not fix them, and the loop learned fewer rules than the harness at about twice the search.
- With LLM proposers, the result is negative in the generated benchmark: Gemma 3 4B and Qwen 2.5 7B matched search-only acquisition (2/18, 4/18, 4/18) and transferred nothing (0/9). A single integration run on six simple functions solved every task with every arm, which shows the pipeline works but not that the library helps.

## Limitations
- Results come from many rounds of changes to code and search. The docs state that search seconds are not comparable across rounds, and most paired comparisons are descriptive with four seeds.
- Grid seeds 0–3 were used for development throughout. The only fresh-seed grid comparison is the frozen-solver run on seeds 4–7 (D45).
- Verdicts are descriptive: no significance tests were run, and the docs say so.
- The "F2" test (search time flat or falling as the library grows) was not run as a per-task curve against library size. Only aggregate search seconds are reported.
- The library effect was bimodal in the first enumeration-only grid runs: it appeared on seeds 0 and 1 only, and the cause was one discovery at the edge of the search horizon.
- Process crashes: an intermittent Windows access violation in the enumerator (exit 139 and similar) recurred in concurrent batches. One control run (seed 2) died four times and is missing from the 90-interaction table.
- Number discrepancy: DECISIONS.md (D45 addendum) gives the machines control as "1.0 rules learned and about 1180 s", while RESULTS.md gives 0.8 and 1196 s. This log uses RESULTS.md. The source of the difference is not recorded.
- The autonomous coding experiment (`CODING-AUTONOMOUS.md`) has protocol and run directories under `runs/coding-autonomous/`, but no results write-up in the docs. Its outcomes are not reported here.
- The coding "LLM-only" arm is a restricted H language, not unrestricted Python generation. Independent tests are finite evidence.
- The worlds (grid, dial panel) and the rule families are the author's own design. Results are not evidence about general program induction.
- A standing list of open items is in RESULTS.md (clamp, wrap, linked, decay-color-clamp, both panel folds).

## Key files
- `RESULTS.md`: measurement rounds 1–8 for the grid and dial curricula, with all paired tables.
- `DECISIONS.md`: design decisions D1–D65, each with its measurement and addenda.
- `opal-architecture-v0.1.md`: the design thesis; section 10 lists the falsification criteria (F1–F5).
- `PLAN.md`: module layout and build steps; status as of 2026-09-07.
- `RESEARCH.md`: literature check (DreamCoder, Stitch, LAPS, LILO, Levin search, LEGO-Prover, and others) run before the build.
- `AGI-ROADMAP.md`: capability targets and milestones; states that no capability is proven.
- `opal-architecture-v0.1-critique.md`: external critique of the thesis, 2026-09-06.
- `opal/agent.py`: solve tasks, refit grammar weights, sleep, reuse and challenge.
- `opal/abstraction.py`: anti-unification abstraction with a description-length gate.
- `opal/search.py`: typed enumeration with observational-equivalence pruning and backward constraints.
- `opal/world.py`, `opal/machines.py`: grid world and dial-panel domain with hidden rule families.
- `opal/split.py`: recursive split-and-solve driver.
- `opal/propose.py`, `opal/dream.py`: LLM and retrieval proposers; dream-trained guide.
- `opal/loop.py`: goal-directed agent loop (D64–D65).
- `opal/coding/benchmark.py`, `opal/coding/autonomous.py`, `opal/coding/learner.py`: coding benchmarks, autonomous investigation, and persistent coding learner.
- `CODING.md`, `CODING-RESULTS.md`: coding learner usage and the six-function integration results.
- `CODING-LEARNING-PROTOCOL.md`, `CODING-LEARNING-RESULTS.md`: frozen generated-rule protocol and results.
- `CODING-AUTONOMOUS.md`: bounded autonomous coding experiment design (no results write-up).
- `tests/`: test modules, one per source module; a full run on Python 3.12 passed 132 tests (CODING-LEARNING-RESULTS.md).

## Sources
- `RESULTS.md` (all grid and dial tables, rounds 1–8)
- `DECISIONS.md` (D32, D40, D44, D45, D47, D48–D50, D53, D55, D58, D59, D60, D62, D63, D64, D65; D43 addendum 6 for hardware)
- `CODING-RESULTS.md` (integration table)
- `CODING-LEARNING-PROTOCOL.md` and `CODING-LEARNING-RESULTS.md` (generated benchmark)
- `CODING.md` and `CODING-AUTONOMOUS.md` (design and commands)
- `opal-architecture-v0.1.md` (section 10), `PLAN.md`, `RESEARCH.md`, `AGI-ROADMAP.md`
