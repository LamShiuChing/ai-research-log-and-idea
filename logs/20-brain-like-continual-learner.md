# Brain-like continual learner: fact memory, slots, sleep on its own dreams, continual GPT
**Dates:** 26–30 Sep 2026 (fact-memory 26–29 Sep; AGI123 29–30 Sep) · **Status:** Paused. The fact-memory project was stopped by the user on 29 Sep 2026; AGI123 was paused on the night of 30 Sep 2026, with nothing running · **Hardware/scale:** One PC with an NVIDIA RTX 4080 SUPER (16 GB) and a CPU. Most toy and GPT runs are small (the final report gives a 3M–39M range). Most results are one seed; the key ones are three seeds.

## Question
Can one learner built along brain-like lines learn at runtime from few examples, one slice at a time, keep what it learned without replaying stored data, and stay a fixed size? The project tested each part on real data and asked which parts hold and where they fail.

## Approach
- **Fact memory (fact-memory/):** slot memories where each slot is a sum over fixed features, solved by least squares; writes only the surprise (delta rule); importance (use counts with fading) decides what survives when memory is full.
- **Running-sum links (AGI123/connect/, AGI123/falsify/):** W = A(B + λI)⁻¹ with A += y xᵀ and B += x xᵀ. Order-free and exact while the features under it stay fixed.
- **Sleep on its own dreams (fact-memory/cgpt.py):** before a night, the old model writes 512 dreams of 256 bytes, started from a newline (later from varied starts), with no stored old text. During the night the model is also asked to keep its old predictions on those dreams (KL to a frozen copy). Weight and dream count were varied in cgpt_gap.py.
- **Continual GPT (cgpt.py):** nanoGPT redesigned, byte level, 4 blocks, width 256, 4 heads, 256-byte context, trained from scratch. Six BabyLM 10M "clean" domains, one per day: 1,500 steps of 32 × 256 bytes per day; each day loads the last checkpoint (warm start), learns, saves, then is tested on every domain's held-out text. A slot variant (1,024 key/value slots, 32 read per position) was also tried.
- **Runtime brain (fact-memory/brainchat*.py):** pretrained gpt2-mini (39M, word pieces) reads one Wikipedia page a day, sleeps (learns until under 2% wrong or no progress for 100 steps, then repairs against its old self), and is probed by a tester's verbatim questions.
- **Puzzle brain (fact-memory/brain.py, brain_stream.py; AGI123/brain/):** a joined system of exact notebook rows, cortex sheets, concept programs, importance and fading, nightly replay and dreams. The v0.7 package adds a manager, a checker (inner world model), a guesser (TD values plus replay), and skills. The guesser's programs use a frozen Qwen2.5-Coder-0.5B as a sense.
- **Frozen senses:** ResNet-18 for CIFAR-100 images; fixed word hashing (4,096 numbers) for 20 Newsgroups text.
- **Datasets:** BabyLM 10M "clean" (CHILDES, Simple Wikipedia, BNC spoken, Gutenberg, QED subtitles, children's stories; also Children's Book Test and Switchboard in one test); MQuAKE-CF-3k; 1D-ARC (901 puzzles); BIG-bench List Functions (250 functions); Boxoban; CIFAR-100 (100 kinds, 5 new per day); 20 Newsgroups (20 topics, 1 new per day); WikiText-103 (first 50M words); SimLex and WordSim; WordNet noun classes; 12.85M Duolingo reviews and TempLAMA (falsifiers); BabyAI (MiniGrid GoTo, PickupDist, DoorKey, UnlockPickup); 13 Wikipedia pages read one a day through the MediaWiki API.
- **Predictions:** written in each script's docstring or report before the run, then graded in the same file. Examples: plain GPT forgetting ≥ 0.10 bits per byte (G1); dreams-GPT forgetting ≤ 1/3 of plain (G2); 20-day link, day-1 classes within 3 points of an all-at-once link (CIFAR and 20 Newsgroups); one sentence learned in a 30-step night on 3/3 seeds (N1); pages still ≥ 6/10 after day 13 (L2). Predictions that failed are listed in the table.

## Results

| Test | Result | Baseline / control | Verdict |
|---|---|---|---|
| Slot memory, 500 one-shot facts (fact_test.py) | 0.92 | dense network 0.20–0.25 | Pass (toy) |
| Sum-based slots with sleep, 35 concepts (slot_logic.py) | 1.00 retention; saved checkpoint learns 10 more with old kept | plain fine-tuning of the checkpoint loses 13% | Pass (toy; keeps all data and retrains nightly) |
| Frozen random features + exact least squares, no stored pages (bounded_sleep.py) | 1.00 | curvature summaries 0.85–0.89 | Pass (toy) |
| Importance-weighted memory when full (importance_weights.py) | recurring facts 0.94–0.97 | 0.78 | Pass (toy) |
| Fact updates with surprise writes (valued_keys.py) | updates 0.93; old facts 0.997; students 0.990 | not stated | Works (toy) |
| Universal channel on MQuAKE-CF-3k (channel.py) | chains 0.99–1.00; edited facts 0.91 (channel_keys.py) | channel adds no error | Pass (toy) |
| Learning-to-learn from a 3,000-rule childhood (learn_to_learn.py) | 0.742 | without childhood 0.731; ideal searcher 0.897 | No few-shot gain |
| List Functions (250): brute force / dream-trained guesser / generic primitives / sub-goals | 90/250 / 88/250 / 122 / 134–143 (21/21 decompositions right) | brute force 90/250 | Sub-goals help; guesser alone no better than brute force |
| Closed-loop practice on unseen List Functions | 45.0 → 45.7 | 45.0 | No gain (noise) |
| 1D-ARC unseen puzzles, joined brain (brain.py) | 144 → 152; 149 after two days of List Functions | 144 at start | Gain from structure; plateau about 150 (final report's figure, "out of 450") |
| Concept frames found automatically (concepts.py) | 151 → 165 with no training; 74% of concept answers right | 151 | Gain from new structure |
| Word meanings from List Functions descriptions (words.py) | 14 of 16 key words right | shuffled control 2/16 | Pass; but hearing words during search cut solves 39 → 27 |
| Brain expressibility on Boxoban and generated worlds (probe_world.py) | 0/100 Boxoban rows; 12/100 generated rows | — | Fail; planned childhood could not start |
| Day-by-day GPT, plain (cgpt.py) | forgetting 0.583; just learned 1.825; final average 2.311 | joint all-at-once ceiling 1.812 | G1 pass (forgetting predicted ≥ 0.10) |
| Day-by-day GPT, sleep on own dreams (cgpt.py) | forgetting 0.205; just learned 1.828; final 1.999 | plain 0.583 | About 35% of plain's forgetting; learns as well as plain |
| Day-by-day GPT, slots + dreams (cgpt.py) | forgetting −0.003; just learned 2.334; final 2.331 | plain 1.825 learned | G2 pass; G3 (learns within 0.05) fail; G4 fail |
| Day-by-day GPT, slots alone (cgpt.py) | forgetting 0.244; just learned 2.293 | plain 1.825 | Learns worse; as built, slots do not help |
| Sleep strength, 3 seeds: weight 1, 512 dreams (cgpt_gap.py) | forgetting 0.204; final 2.003 | — | Rerun baseline |
| Sleep strength, 3 seeds: weight 3 | forgetting 0.137; just learned 1.840; final 1.954 | weight 1 | D3 (< 0.12) fail |
| Sleep strength, 3 seeds: 2,048 dreams | forgetting 0.168; just learned 1.831; final 1.971 | weight 1 | Not predicted |
| Sleep strength, 3 seeds: weight 3 + 2,048 dreams | forgetting 0.083; just learned 1.854; final 1.923 | plain 0.58 / 2.31; joint ceiling 1.81 | D1 (final ≤ 1.92) missed by a hair; D2 pass. Best of four arms, chosen after the grid ran (post hoc) |
| One sentence in one night, 3 seeds (cgpt_fact.py) | 30 rehearsals: "today is 9/29/2026." completed on 3/3; 1 rehearsal 0/3; kept after 3 later nights 2/3; update to 9/30/2026 on 3/3 | no dreams, 30 steps: ordinary text +0.78 to +0.92 bits/byte | N1, N3, N4 pass; with dreams the 30-step cost was 0.03 (predicted ≤ 0.01) |
| Varied dream starts (cgpt_dreams2.py) | dreams holding the date after a 30-step night: 24–40 of 512 (was 497–509); kept after 3 later nights on 3/3 | old dreams: kept 2/3 | Obsession removed; R2 (≤ 20 of 512) and R3 (30-step cost ≤ 0.02) failed; R4 and R5 pass |
| Hippocampus only, no night (cgpt_fact.py) | completes the date on 3/3 | — | N6 pass |
| 13-day Wikipedia stream, gpt2-mini brain (brainchat_stream.py) | page learned the morning after its night: 7–10 of 10 (Tunguska 7); after day 13, pages from days 1–5 at 2, 1, 3/5 (stories), 5, 5 | predictions ≥ 8/10 (L1) and ≥ 6/10 (L2) | L1 failed narrowly; L2 failed; L3 failed: general English 6.34 (day 3) → 6.85 (day 13) |
| 13-day stream, plain fine-tuning baseline | not run | — | How much the nights help is not measured |
| 20-day CIFAR-100, running-sum link vs 2-layer net (twenty_days.py) | day-1 kinds at day 20: link 0.79, net 0.00; all kinds at day 20: link 0.60, net 0.05 | link all at once 0.60; net all at once 0.58 | Link keeps old kinds; net forgets. Predictions 2 and 3 pass; prediction 1 recorded as pass, see Limitations |
| 20-day 20 Newsgroups, running-sum link vs net (twenty_days.py) | day-1 topics at day 20: link 0.62, net 0.00; all topics: link 0.60, net 0.05 | link all at once 0.60; net all at once 0.54 | Pass on all three predictions; link day by day equals all at once up to float32 rounding |
| Meaning from context, WordSim, WikiText-103 (decisive/REPORT.md) | B (running totals) 0.54; C (skip-gram) 0.36; D (batch reference) 0.57 | A (spelling codes) 0.05; prediction for C was 0.55 (wrong) | Meaning enters from context codes |
| Meaning, SimLex | B 0.30; C 0.22; D 0.28 | A 0.04 | — |
| New cases: naming WordNet classes of unseen nouns, top-1 | B 0.159 (2.15× A) | A 0.074; majority baseline 0.233 | Kill rule verdict: inconclusive; the 3× bar was not met |
| Memory with re-keying, 600 invented facts, 7 nights (decisive) | recall 1.000 for A, B and C; paraphrase recall B 0.76, A 0.53 | unlimited archive recall 1.000 | Re-keying keeps memory; B better on paraphrase |
| Block code capacity, N = 1024 (unit/RESULTS.md P1) | block ~1,500 pairs vs dense ~300 (0.91) | predicted ≥ 10× | Failed (about 5×) |
| Two senses meet in one code, per-block ports (P3) | eye 0.41, ear 0.40; agree 0.19 | predicted ≥ 0.9 | Failed. Whole-code variant (0.98 / 0.97) used privileged information and does not count |
| Falsifier F1: running-sum links when the lower area learns | day-1 digits 1.00 → 0.00; changed facts slow; cannot do "same?" | — | Partly broken |
| Falsifier F2: pulse code as meaning container | "dog bites man" = "man bites dog"; 18.6% false matches at 10⁶ sentences | — | Broken as container; fine as address |
| Falsifier F3: replay protecting learning areas | plain replay; keep rule kept mostly the first days | — | Broken |
| Falsifier F4: hand-set keep rule and confidence on 12.85M Duolingo reviews | loses to tuned ACT-R and LRU; said 50% when recall was 90% | tuned ACT-R / LRU | Broken |
| Falsifier F5: hub from co-occurrence (real images and spoken digits, no labels) | 0.535 cross-sense | chance 0.10 | Survived weakly; needs ~30k pairs; fails on shifts |
| Falsifier F6: manager value table and gate | no transfer to new goals | — | Broken |
| BabyAI, 5-day brain, full (brain/README.md) | Dist 0.65 (seeds 0.58, 0.72); I1-like 0.59 | ablations: no slots 0.02; no guesser 0.02; no checker 0.00; no night 0.56; no skills 0.59; always habit (no confidence, 1 seed) 0.72 | Slots, guesser and checker are needed; skills and confidence earned nothing; always-habit scored higher (one seed) |
| Partial view, seed 2 forgetting (brain/README.md) | UnlockPickup 1.00 → 0.17 after day 5 (diagnosis); the guesser alone caused it | day-4 brain 1.00 | Diagnosed. room_key fix: 1.00 on seed 2 (partial-view Dist falls from .58/.38 to .30/.26) |
| Count-based guesser, partial view (brain/README.md) | Dist 0.36 on both seeds; day-5 training .46 / .52 | table guesser Dist .30 / .26 | Better on Dist and day-5 training. One unexplained seed-2 failure on UnlockPickup |
| Life test, 5 seeds: skill recall (brain/README.md) | practised .23 ± .04; unpractised .25 ± .06; newborn .05 | B3 adaptive + LRU (3 seeds): .30 / .26 / .27 | Practice did not help |
| Life test, 5 seeds: new List Functions S@10 (birth → end) | .00 → .05 ± .02 | — | Grows little; S@100 .04 → .12 |
| Life test, 5 seeds: facts (brain/README.md) | R_imp .88 ± .02; update .86 ± .01 | B3: R_imp .51; update .55 | Facts kept; but not a fair win (see Limitations) |
| Life test, 5 seeds: look-up (LQ) | −0.23 ± .04 (AUROC .77 ± .03) | B3: LQ .76 | Worse than never looking up, on every seed |
| Life test, 5 seeds: size (256 kB budget) | 1.52 ± .01, growing 0.35 per 10 days | B3: .06 | Over budget on every seed |

## What it showed
- **Exact memory only for sums over fixed features.** The 1.00 results came from least squares on frozen features. Learners whose features also learned forgot to some degree (plain GPT: 0.583 bits per byte, averaged over the domains). Every fix was partial.
- **Sleep on its own dreams cut forgetting, but did not remove it.** Dreams alone took the day-by-day GPT from 0.583 to 0.205 bits per byte at the same learning. The best of four sleep settings reached 0.083 on three seeds, but that setting was chosen after the grid ran and missed its own predicted final average by a hair. Older pages in the 13-day Wikipedia stream still fell from 7–10 of 10 toward 1–5 of 10.
- **Running-sum links kept old classes on 20-day streams.** Day-1 CIFAR-100 kinds stayed at 0.79 at day 20 and day-1 newsgroup topics at 0.62, while the normal network fell to 0.00. These match an all-at-once link. The link breaks when the layer under it learns (falsifier F1: 1.00 → 0.00).
- **Practice and invention added little on their own.** Closed-loop practice gave 45.0 → 45.7. Skill recall was no better when practised (.23 vs .25). Gains came from structure added by hand, not from practice.
- **Final verdict: too shallow.** The user judged the findings too shallow. Each experiment reduced forgetting, but none removed its cause. In a network whose shared weights hold everything it knows, learning new things moves weights that old knowledge depends on. The one open lead: is there a learning rule or architecture in which deep networks keep the "sums do not overwrite" property?

## Limitations
- Most results are one seed; the key ones are three. Models are small (toys, a 39M gpt2-mini, and an 85M GPT stopped at day 3).
- The 0.083 arm was the best of four sleep settings on three seeds, chosen after the grid ran (post hoc). The pre-registered single-seed result is 0.205.
- The 13-day stream used one tester's 125 verbatim probes, has no plain fine-tuning baseline, and general English drifted 6.34 → 6.85.
- The dream arms learned slightly worse than plain (1.854 vs 1.825–1.834 in cgpt_gap.py). The final report's phrase "without slowing learning" is looser than the table.
- Several striking results were withdrawn when privileged information was removed: one-shot words, fact sorting, fact layers, and two senses meeting in one code. They do not count.
- The "cuts forgetting 7×" statement compares a best-of-grid arm with a single plain run.
- Source inconsistencies, recorded as found:
  - CIFAR-100 prediction 1 (day-1 kinds within 3 points of an all-at-once link) is recorded as PASS, but 0.79 is not within 3 points of 0.60. The wording is ambiguous.
  - Partial-view UnlockPickup on seed 2 is 0.17 in the diagnosis and 0.20 in the table.
  - The brain test count is 28 in brain/README.md and 25 in HANDOFF.md.
  - The 1D-ARC plateau is "about 150/450", while the dataset has 901 puzzles; the denominator for the unseen set is not stated.
- Float32 features broke the exact-equality check (B differs by 8e-5 and W by 3.7e-7, relative). The docs treat this as rounding.
- The meaning test's 3× bar was not checked for reachability before running; the decisive report calls it a flaw found after the run. The kill rule is inconclusive.
- The life test's "facts beat B3" result is due to nothing being dropped, which broke the size budget. It is not a fair win.
- AGI123/MISSING.md is a gap list (about 23 of 47 parts missing), not a result.

## Key files
- `fact-memory/FINAL_REPORT.md`: full report of the fact-memory project (26–29 Sep), with the 13-day stream table and the verdict.
- `fact-memory/BRAIN_RESEARCH_SUMMARY.md`: neuroscience findings the project used, with how each was checked and what was taken from it.
- `fact-memory/cgpt.py`: continual GPT with slots and dreams; the day-by-day protocol, predictions G1–G4, and the 0.583 / 0.205 results.
- `fact-memory/cgpt_gap.py`: sleep-strength grid (weight, dream count), predictions D1–D3, and the 0.083 arm.
- `fact-memory/cgpt_fact.py`: one sentence learned in one night, with predictions N1–N6.
- `fact-memory/cgpt_dreams2.py`: varied dream starts and key-value cache; predictions R2–R5.
- `fact-memory/brainchat.py`, `brainchat_stream.py`: runtime brain on gpt2-mini; 13-day Wikipedia stream with predictions L1–L3.
- `fact-memory/slot_logic.py`, `bounded_sleep.py`, `importance_weights.py`, `learn_to_learn.py`: memory toys (exact sums, importance, few-shot learning).
- `fact-memory/brain.py`, `brain_stream.py`: joined puzzle brain on 1D-ARC and List Functions.
- `fact-memory/concepts.py`, `words.py`, `probe_world.py`: concept frames, word meanings, and the expressibility probe.
- `AGI123/connect/twenty_days.py`: 20-day CIFAR-100 and 20 Newsgroups, running-sum link vs network; predictions and results in its docstring.
- `AGI123/decisive/decisive.py`, `REPORT.md`: meaning from counting on WikiText-103, with the kill rule and predictions.
- `AGI123/falsify/SUMMARY.md`: six falsifiers (F1–F6) and their verdicts.
- `AGI123/unit/RESULTS.md`, `sim_unit.py`: block-code capacity, binding, and two-sense tests (P1–P3).
- `AGI123/brain/brain.py`, `README.md`, `programs.py`, `run_babyai.py`, `run_life.py`: the v0.7 package, its ablations, and the 5-seed life test.
- `AGI123/HANDOFF.md`, `AGI123/MISSING.md`: the AGI123 handoff (evidence table, open problems) and the gap map.

## Sources
- `fact-memory/FINAL_REPORT.md` (identical copy in `AGI123/FINAL_REPORT.md`)
- `fact-memory/BRAIN_RESEARCH_SUMMARY.md`
- `AGI123/HANDOFF.md`, `AGI123/MISSING.md`
- `AGI123/brain/README.md`
- `AGI123/falsify/SUMMARY.md`
- `AGI123/decisive/REPORT.md`
- `AGI123/unit/RESULTS.md`
- Module docstrings: `fact-memory/cgpt.py`, `cgpt_gap.py`, `cgpt_dreams2.py`, `cgpt_fact.py`, `brainchat_stream.py`; `AGI123/connect/twenty_days.py`
