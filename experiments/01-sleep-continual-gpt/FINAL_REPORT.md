# Final report: the continual-learning brain (fact-memory), 26-29 September 2026

Status: **stopped by the user on 29 September 2026.** Last commit of the work: `480ebbf`.
Every script keeps its predictions (written before running), results and verdicts in its docstring; raw numbers are
in `results/`. This report summarises the whole project; the details are in the handoffs
(`HANDOFF_2026-09-27.md`, `HANDOFF_2026-09-28.md`, `HANDOFF_2026-09-28b.md`) and session logs.

## 0. Summary

The goal was one continual learner, built like a brain: it learns at runtime from few examples, one slice at a
time, keeps what it learned without replaying stored data, stays a fixed size, can be saved and reopened, reuses
what it learned to learn more, and understands language.

What was achieved:
- Memories that are **exact and never forget**, but only for sum-based (least-squares) memories on **fixed**
  features: facts, puzzle-step ranking, concepts (1.00 retention in several toys).
- A puzzle-solving brain (`brain.py`) that joins hippocampus, cortex sheets, importance, fading and sleep, and keeps
  what it learned across days (1D-ARC unseen puzzles 144 -> 152, kept at 149 after switching to another dataset).
- For a GPT, **sleep on its own dreams** cuts forgetting about 7x (0.58 -> 0.08 bits per byte) without slowing
  learning, and a **new sentence or page is learned in one short night** (10 rehearsals, 0.1 s; a page in ~1
  minute) and can be updated the next night.

What was not achieved:
- **Long-term keeping in a learning network.** Over a 13-day stream of Wikipedia pages, each page was learned the
  night it was read (7-10 of 10 probes), but pages read 10 days earlier fell to 1-5 of 10, and general English wore
  down every night.
- **Using what it learned to learn more.** Gains always came from structure we added, not from practice.
- **An honest fast memory** (hippocampus) that recalls anything useful.
- **Meaning and language** beyond word-to-step associations.

The user's conclusion: the findings are too shallow; the root problem (shared, dense weights forget) needs
fundamental research on the mathematics of neural-network learning, or a new architecture.

| Question | Best honest answer | Where |
|---|---|---|
| Can a memory keep everything? | Yes, exactly (1.00), for sums on fixed features | `bounded_sleep.py`, `slot_logic.py` |
| Can importance decide what survives when memory is full? | Yes: recurring facts 0.94-0.97 vs 0.78; one-offs pay | `importance_weights.py` |
| Can a GPT learn day by day without forgetting? | Much less forgetting (0.58 -> 0.08), not none | `cgpt.py`, `cgpt_gap.py` |
| Can it learn one sentence in one night? | Yes: 10 rehearsals; kept 3/3 after 3 nights | `cgpt_fact.py`, `cgpt_dreams2.py` |
| Does it keep pages over two weeks? | No: 1-5 of 10 after 10 days | `brainchat_stream.py` |
| Can the puzzle brain get better by practice? | No: gains only from new structure | `loop.py`, `brain_stream.py` |

## 1. Goal and working rules

The user's vision (from the start of the project): a learner with a fixed or stable number of units that keeps
learning like a brain; a deployed checkpoint that is used read-only and later reopened to learn more; memories in
the hippocampus that fade once slow memory holds them, least important or oldest first; not replay of stored data.
Later the user added: language is the door to meaning in the human world; motivation can wait.

Working rules that held throughout:
- Reason from first principles and our own data; papers are supporting evidence only.
- Use real datasets (Hugging Face and others) instead of inventing data.
- Write predictions before each run; label anything decided afterwards as post hoc; report failures plainly.
- **No privileged information** (added 29 Sept): data may come in any form, and architecture and settings are
  free, but the learning machinery may not use anything the model could not have at runtime (answer boundaries,
  fact identities, hand-picked episodes, labels in the stream).

## 2. What was done, in order

### Phase 1 (26 Sept): fact memory in a small world
- Review of 25 earlier projects (about 3,000 documents): simple baselines beat complex models, and every
  architecture failed on unseen combinations.
- `fact_test.py`: a slot memory keeps 500 one-shot facts where a dense network keeps 0.20-0.25 (0.92 vs 0.20-0.25).
- `best_bet.py`, `week_test.py`: a neuroscience-style system (gated writes, tagged sleep, an address map) in a
  7-day world of 100 people with 60 fact changes. With hand-set "same fact?" rules, 82-90% of changes on messy text
  were never linked. `h3.py`: learning sameness from corrections raised changed facts to 0.71-0.91.

### Phase 2 (27 Sept): a new memory design and the first milestones
- `update_probe.py` showed an earlier "the memory half works" claim was wrong: the cortex never updated anything by
  itself (new value 0.19-0.23 on raw replay).
- New design: belief = cortex + fast surprise slots; write only the surprise (delta rule); sleep re-solves the slots
  so the sum is unchanged. `valued_keys.py`: updates 0.93, old facts 0.997, students 0.990. `continual.py` (28
  days): the online key map keeps improving (0.969); dropping absorbed slots is catastrophic (0.35).
- `importance.py`: a dense cortex under a replay budget collapses to about 0.55 whatever decides the replay.
  `net_week.py`: a structured cortex replaying by its own error reaches 0.82 at a quarter of the rows.
- Milestones: **M1** universal channel on MQuAKE-CF-3k (`channel.py`: chains 0.99-1.00, the channel adds no
  error; `channel_keys.py`: edited facts 0.91). **M2** learned gate (`gate_keep.py`: 0.973 seen, 0.880 unseen
  combinations). **M3** skills (`skills.py`: 0.13, failed because memory never says "no such fact"); `menu.py`:
  0.98 with 4x fewer reads.

### Phase 3 (28 Sept): bounded memory, reasoning, invention, one joined brain
- Toys: `slot_logic.py` / `slot_stages.py` / `slot_resume.py`: slots + sleep keep 35 concepts at 1.00 and a saved
  checkpoint reopens and learns 10 more with the old kept (plain fine-tuning of the checkpoint loses 13%); but this
  is keep-all-data plus nightly retraining. `bounded_sleep.py`: frozen random features + exact least squares keep
  everything with no stored pages and a constant nightly cost; curvature summaries only 0.85-0.89.
  `importance_weights.py`: weight = uses keeps recurring facts 0.94-0.97. `drift.py`: when lower features drift,
  translating the summary with enough anchor questions keeps 1.00. `learn_to_learn.py`: a 3,000-rule childhood
  gives no few-shot gain (0.742 vs 0.731); an ideal searcher gets 0.897, so understanding from few examples is
  search over combinations, not feature weighting.
- Reasoning on List Functions (BIG-bench, 250 functions): brute force 90/250; a dream-trained neural guesser 88/250
  in about 2 tries; generic primitives 122; sub-goals 134-143 (21/21 decompositions right). Closed-loop practice:
  no gain on unseen functions (45.0 -> 45.7, noise). Gains only ever came from new structure.
- 1D-ARC (901 puzzles): invention of "ways of seeing" from a grammar kept by what explains failures (object cut
  found every time); an invented idea pays only once the chooser learns from the cases it solved (+44 to +60 on 3
  seeds); more ideas without fading dilute the chooser.
- `brain.py` / `brain_stream.py`: **one joined continual brain** (byte states, keys from a change encoder, concepts
  = programs; hippocampus, cortex sheets, concept sheet, importance + fading + caps, facts, nights, save and load).
  1D-ARC unseen 144 -> 151 -> 152, 149 after two days of List Functions; 49-73% of right answers from memory; the
  reloaded brain is identical. Plateau about 150/450: the brain cannot express colour slots or local rules.
- `probe_world.py`: the brain could express 0/100 Boxoban rows and 12/100 generated world rows, so the planned
  "childhood" could not start.

### Phase 4 (28-29 Sept night): ranking, concepts, words, architecture
- `rankers.py`, `rankers2.py`: ranking candidates by progress is best (List Functions 124/250, 1D-ARC 116/901); a
  shortlist of 100 keeps most of that at 12% of the cost.
- `concepts.py`: frames with blanks found automatically ("cut into objects, do ? to each"): 151 -> 165 with no
  training; 74% of concept answers right (one coincidental frame gave all 4 wrong answers).
- `words.py`: word meanings learned from List Functions' own descriptions paired with the brain's own programs: 14
  of 16 key words right (shuffled control 2/16), no labels. But hearing the words during search hurt (39 -> 27):
  a word lifted every variant of its step and pushed out the eyes' right guess.
- Architecture discussion with the user: language as a door into one shared brain plus an inner voice; thinking in
  concepts, language as a tool on top; motivation on hold; the soft side of the cortex to be a continual GPT.

### Phase 5 (29 Sept): the continual GPT and the runtime brain
- `cgpt.py` (nanoGPT redesigned, byte level, BabyLM 10M, 6 domains, one per day, checkpoint -> test -> warm start):
  plain GPT forgets 0.58 bits per byte; **sleep on its own dreams** (keep old predictions on 512 dreams) 0.20 with
  the same learning; `cgpt_gap.py`: weight 3 and 2,048 dreams 0.08 (ceiling of learning everything at once 1.81 vs
  1.92). A slot cortex replacing the MLP learned worse (2.29 vs 1.83).
- `cgpt_sheet.py`: a tally-sheet read-out does not forget but predicts about 0.8 bits worse (least squares gives
  poor probabilities).
- One sentence in one night (`cgpt_fact.py`, `cgpt_why.py`, `cgpt_dreams2.py`): 10 rehearsals learn "today is
  9/29/2026."; without dreams the rest of the model is damaged (+0.4 to +1.0); with dreams a longer night is gentler
  (100 steps +0.006). Dreams that always began at a newline let the sentence take over ~500 of 512 dreams; varied
  starts fixed it and the date then survived 3 later nights on 3/3 seeds. A key-value cache cut 512 dreams from 33.7
  s to 0.8 s.
- `cgpt_surprise.py`, the honest fast memory (writes only surprises keyed by the model's own state; settings chosen
  by its own error): no gain on new words, almost no fact recall; nights rehearsing what came back unchanged got
  stable facts to 0.50.
- `brainchat.py`, the runtime brain: `feed` (read in slices, keep surprises and cues), `sleep` (learn until under 2%
  wrong or no progress for 100 steps, then repair while the drift from the old self falls; half the dreams start
  from cues of earlier reading), `ask`, checkpoints. First on the small byte model, then on **pretrained gpt2-mini**
  (39M, word pieces; our code reproduces its top guesses at 100%).
- Wikipedia runtime tests with a tester subagent writing verbatim probes: 3 pages (gpt2-mini keeps Wirksworth 9/10
  after two more days vs 3/10 for the byte model with fixed nights), then a 13-day stream (below).

## 3. The final test: a 13-day Wikipedia stream (gpt2-mini brain)

Probes right the morning after each night (10 per page; the stories 5):

| Page (day read) | 4 | 6 | 8 | 10 | 12 | 13 |
|---|---|---|---|---|---|---|
| Wirksworth (1) | 7 | 8 | 5 | 6 | 4 | 2 |
| Axolotl (2) | 6 | 6 | 2 | 2 | 1 | 1 |
| Children's stories (3) | 5 | 5 | 4 | 4 | 3 | 3 |
| Olm (4) | 8 | 7 | 6 | 7 | 6 | 5 |
| Tristan da Cunha (5) | | 6 | 6 | 5 | 5 | 5 |
| Voynich manuscript (6) | | 9 | 9 | 8 | 8 | 7 |
| Antikythera mechanism (7) | | | 9 | 9 | 8 | 8 |
| Great Stink (8) | | | 10 | 10 | 8 | 7 |
| Kowloon Walled City (9) | | | | 8 | 8 | 7 |
| Svalbard Global Seed Vault (10) | | | | 10 | 10 | 10 |
| Hagfish (11) | | | | | 10 | 10 |
| Carrington Event (12) | | | | | 8 | 7 |
| Tunguska event (13) | | | | | | 7 |
| General English (bits per word piece, lower is better; 6.34 on day 3) | 6.42 | 6.54 | 6.57 | 6.61 | 6.76 | 6.85 |

Each page is learned in a 40-60 second night that stops itself; forgetting is gradual and by age, but it compounds.
A plain fine-tuning baseline for this stream was not run, so how much our nights help here is not measured.

## 4. Results that do not count

These used information the model could not have at runtime and are labelled in their files:
- `cgpt_hippo.py`, `cgpt_hippo2.py`: new words learned from one meeting (64-70% better, 82-90% kept after sleep),
  but the hippocampus held windows we chose around each new word.
- `cgpt_sort.py` (use-count and sorting arms), `cgpt_factlayer.py`, `cgpt_sharp.py`: our question / answer split
  and our fact identities were used. The consolidate-all arm of `cgpt_sort.py` used none and stands (facts
  0.93-1.00, ordinary text +0.19).
- Earlier: `guesser2.py`'s blocks were chosen after reading the dataset's descriptions (the later generic base
  removed most of this); the "who" tag in the fact-memory work was hand-given; some puzzle primitives were designed
  after looking at the test tasks.

## 5. What we learned

1. **Order-free sums do not forget; shared learned weights do.** Every exact result (1.00) came from sums over fixed
   features (least squares, slots re-solved against a fixed cortex). Every learner whose features also learn
   forgot to some degree, and every fix was partial.
2. **Rehearsing the model's own dreams is the strongest general protection we found** (about 7x less forgetting),
   but it protects only what the dreams happen to show; specific facts outside them drift a little every night.
3. **How hard a night pushes matters more than how much is read.** Drilling one small page 200 times overwrote
   earlier pages; stopping when the page is known, and repairing while today's text and the dreams train together,
   was gentler. Dream-only repair undid the new learning.
4. **A fast memory must separate patterns and still generalise; our honest versions did one or the other.** Strict
   state matching never fires in a new context; loose matching recalls wrong words.
5. **Importance is zero-sum when memory is full, and counting uses is not enough**: a question asked daily with a
   changing answer ("today is ...") would win on uses but should stay in fast memory.
6. **Invention without new kinds of pieces plateaus.** The puzzle brain found useful combinations of its grammar,
   and practice alone added nothing.
7. **Honesty checks changed the conclusions.** Several striking results (one-shot words, fact sorting, fact layers)
   disappeared once the machinery was limited to what the model itself could know.

## 6. Why the project stopped, and a direction for fundamental research

The user judged the findings too shallow: each experiment reduced forgetting but none removed its cause. In a
network whose weights are shared by everything it knows, any update for new knowledge moves weights that old
knowledge depends on; rehearsal, sparsity and importance only limit how far.

The one lead from our own data: learning was exact and order-free only when it could be written as adding to sums
(A = sum phi phi^T, b = sum phi y) over features that stay fixed. A fundamental question for next time: is there a
learning rule, or an architecture, in which deep, feature-learning networks keep that "sums do not overwrite"
property, so that learning something new cannot move what was learned before?

## 7. Code and data map (fact-memory/)

- Fact memory: `fact_test.py`, `best_bet.py`, `week_test.py`, `h3.py`, `delta.py`, `learned_keys.py`,
  `valued_keys.py`, `continual.py`, `importance.py`, `net_week.py`, `channel.py`, `channel_keys.py`, `gate*.py`,
  `skills.py`, `nothing.py`, `menu.py`.
- Memory toys: `sparse_logic.py`, `slot_logic.py`, `slot_stages.py`, `slot_resume.py`, `bounded_sleep.py`,
  `importance_weights.py`, `drift.py`, `learn_to_learn.py`.
- Reasoning and invention: `list_search*.py`, `guesser*.py`, `lowlevel.py`, `subgoals*.py`, `loop.py`, `arc1d.py`,
  `cuts.py`, `invent_*.py`, `rankers*.py`, `concepts.py`, `words.py`.
- The joined brain: `brain.py`, `brain_stream.py`, `world.py`, `probe_world.py`.
- Continual GPT: `cgpt.py` (bytes or GPT-2 word pieces; `set_size`), `cgpt_gap.py`, `cgpt_sheet.py`, `cgpt_fact.py`,
  `cgpt_why.py`, `cgpt_dreams2.py`, `cgpt_surprise.py`, `cgpt_big.py` (own-grown 85M, stopped at day 3), and the
  flagged `cgpt_hippo*.py`, `cgpt_sort.py`, `cgpt_factlayer.py`, `cgpt_sharp.py`.
- Runtime brain: `gpt2mini.py` (loads huggingface.co/erwanf/gpt2-mini), `brainchat.py`, `brainchat_probe.py`,
  `brainchat_stream.py`. Brains live in `results/checkpoints/brainchat/<BRAIN>/` (git-ignored).
- Data (git-ignored; sources in the handoffs and docstrings): `data/babylm/` (BabyLM 10M "clean", cambridge-climb/
  BabyLM), `data/mquake/`, `data/list_functions/`, `data/1d-arc/`, `data/boxoban/`, `data/brainchat/` (Wikipedia
  pages via the MediaWiki API, probes and probe answers).

Caveats: most toys and many GPT runs are small (3M-39M numbers, thousands to millions of bytes); many results are
one seed, the key ones three; the Wikipedia probes are 125 verbatim completions written by one tester.
