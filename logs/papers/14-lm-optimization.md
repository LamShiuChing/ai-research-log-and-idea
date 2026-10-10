# Length Extrapolation and Composition in Small From-Scratch Transformers: Two Diversity and Positional Effects

**Author:** Eric Lam (independent researcher, Toronto)

## Abstract
We train GPT-style decoders of 5.8M to 88.2M parameters from scratch on a mixture of TinyStories prose and synthetic reasoning tasks written as text traces. We ask two questions. First, when does a model that learned a procedure on short inputs keep working on longer ones? Second, when does it compose two procedures it has seen separately into a pair it has not seen? For length extrapolation, rotary positions gave 0.00 accuracy at 6-digit addition at every size tested in a prose stream, while a model with no positional encoding and a per-document attention mask reached 0.74 at 6 digits. For composition, the limit was not the number of parts but the number of partners each part had in training. On held-out English pairs, composition rose from 0.08 with seven trained pairs (35 held out) to 0.81 with 34 trained pairs (8 held out). Results come from one run per condition in most cases, so they are reported as measured effects with their spread, not as settled laws.

## 1. Introduction
Small models can be trained cheaply, so it is worth knowing which design choices let them generalise beyond their training distribution. Two failures are common. Transformers often do not extrapolate in input length. Models also tend to reproduce memorised combinations of skills rather than composing parts they have learned. We study both in a controlled setting where every model is trained from scratch, so any result can be traced to a choice we made. The aim is a small, reproducible set of results on these two questions, not a benchmark score.

We make three contributions. (i) A length-extrapolation comparison across position schemes and training mixes, including the finding that no-position models with per-document attention extrapolate in a prose stream where rotary models do not. (ii) A series of composition experiments in which the held-out fraction and the number of successors per stage are varied, showing that composition failures track diversity in the training data. (iii) A note on evaluation: several early results were revised once evaluator defects were found, and the revisions are reported in full.

## 2. Method

### 2.1 Models
Decoders follow the GPT-2 recipe: pre-layer normalisation, GELU MLP, tied embeddings. Four sizes were used: 5m, 14m, 38m and 85m. Parameter counts depend on the vocabulary. With a 256-symbol byte vocabulary the sizes are 4.9M, 14.5M, 38.2M and 85.6M; with a BPE-4096 vocabulary they are 5.8M, 15.8M, 39.9M and 88.2M. Training used AdamW with warmup and cosine decay, batch 64 by 512 tokens for stream runs and 128 for single-task runs.

### 2.2 Data and tokenisation
Prose is TinyStories. Reasoning documents are synthetic: addition, multiplication, parity, reachability on random graphs, compare, running maximum, count, bracket matching, two- and three-stage compositions, and nested calls such as `cmp(add(a,b),c)`. Each document is a prompt, a trace and an answer. Numbers and operators are kept as single symbols (a "symbol-preserving" codec). Early work showed that tokenised numbers were the cause of failure: with a word tokenizer, 5-digit addition reached 0.04 to 0.09, against 0.81 to 0.99 with symbol units (F10).

### 2.3 Trace format
A trace is a sequence of steps. The best format writes each step's inputs and state, for example `8+6+0=4c1` for one column of addition (F11). Accumulated histories are excluded, because restating the visited set or the partial sum hurt generalisation (F8). Narration in English was 20 to 40 times longer and did not generalise better.

### 2.4 Positions and attention
Three schemes were compared: learned absolute positions, no positions, and rotary positions. Rotary and learned positions counted columns by position and stopped at the trained length. Without positions, a model counts only from the start of what it attends to. In a prose stream, a document at a random offset has no anchor, so a no-position model needs per-document attention to be trained in a mixed stream (F13).

### 2.5 Evaluation
Accuracy is exact match on fresh problems at the training lengths and at one and two lengths beyond. Composition is tested by holding out specific pairs (or chains) from training and evaluating them under the same protocol.

## 3. Experiments and Results

### 3.1 Length extrapolation
**Table 1.** Accuracy beyond the trained length, addition.

| Setting | Model | 6 digits | 7 digits | 8 digits |
|---|---|---|---|---|
| Single length, 5 digits (F4) | any position scheme | 0.00 | 0.00 | 0.00 |
| Mixed 1–5 digits, CoT (F4) | no positions | 0.86 | 0.03 | 0.00 |
| Mixed 1–5 digits, CoT (F4) | rotary | 0.00 | 0.00 | 0.00 |
| Step format, single task (F11, 3 seeds) | no positions | 0.96 ± 0.01 | 0.01 | – |
| Stream, step format (F11) | rotary, 5m | 0.00 | 0.00 | – |
| Stream, step format (F12) | rotary, 15.8M / 39.9M / 88.2M | 0.00 | 0.00 | – |
| Stream + per-document mask (F13) | no positions, 5m | 0.74 | 0.00 | – |

The table shows the main pattern. Under rotary positions, extrapolation stayed at 0.00 at every size from 5.8M to 88.2M (F12). Scaling improved prose (0.644 to 0.519 bits/byte) and in-range reasoning, but not length extrapolation. Under no positions with a per-document mask, the stream-trained model reached 0.74 at 6 digits, below the single-task no-position result of 0.96 (F11). Seven digits stayed near zero in every condition.

The training range also sets how far a procedure carries. Addition trained on 1–8 digits reached 1.00 at 8 and 9, 0.78 at 10, and 0.01 at 11 (F14). Reachability trained on 5–14 nodes reached 0.85 at 16, 0.77 at 18 and 0.67 at 20. Training on a wider range moved the ceiling; it did not remove it.

### 3.2 Composition
A composition chains two or three primitive procedures, with an intermediate answer written out and a plan header naming the stages. Table 2 summarises the composition experiments. Each row varies the number of trained partners per stage.

**Table 2.** Held-out composition results. Accuracy at the lengths given in each source finding.

| Setting | Trained partners | Held-out result | Source |
|---|---|---|---|
| addcmp, one prompt | composition trained | 0.05 / 0.09 / 0.17 (held out) vs 1.00 / 0.95 / 0.84 (trained) | F15 |
| count.cmp, v5 (four successor types) | two same-shaped successors per first stage | stage-2 name "count" in 40 of 40 samples; never "cmp" | F17 |
| count.cmp, six successors | 6 successors | 0.08 / 0.10 / 0.12; name cmp 15, count 15 of 30 | F18 |
| four held-out pairs, six first stages, seven successors | 38 of 42 pairs trained | sumd.find 1.00 / 0.99 / 0.95; inc.count 0.95 / 0.81 / 0.73; runmax.inc 0.29 / 0.22 / 0.15; count.cmp 0.00 | F19 |
| three-stage chains, 30 trained | sparse links | count.inc.cmp 0.24 / 0.33 / 0.31; add.count.find 0.02 / 0.02 / 0.00 | F20 |
| three-stage chains, 105 trained, every link under three first stages | dense links | count.inc.cmp 0.98 / 0.82 / 0.75; add.count.find 1.00 / 0.95 / 0.84; sumd.rev.cmp 1.00 / 1.00 / 0.79 | F22 |
| English requests, 7 pairs trained, 35 held out | 7 | 0.08 (trained phrasing), 0.14 (held-out phrasing) | F27 |
| English requests, 34 pairs trained, 8 held out | 34 | 0.81 (trained phrasing), 0.57 (held-out phrasing) | F29 |

Three observations follow. First, trained compositions are learned as well as primitives: trained pairs reached 0.95 to 1.00 two lengths beyond training (F16), and trained chains averaged 0.89 to 0.90 (F20, F22). Second, the failure is one of selection, not data flow. Once the held-out pair was made to appear with several first stages, the model copied the second stage's name from the plan rather than looking up a memorised pair (F17 to F19). Third, the same lever worked in English. With seven trained pairs the held-out pairs scored 0.08. With 34 trained pairs, the same kind of held-out pair scored 0.81 (F29), and the model wrote the plan names from the parts on 65 to 100 of 100 samples for seven of eight pairs.

Two further results bound the claim. Copying an operand out of a nested call extrapolates only as far as the copy was trained: a two-operand copy reached 95 of 100 at 10 digits after training to 8, against 1 of 100 at 8 digits when trained to 6 (F23). Three operands in one call remained hard (42 of 100 one step beyond training). And composition cost capacity: adding 109 chain types to a 5m model cost the primitives about 8 points on average (F22).

### 3.3 Interpretation
Both failure types look like lookups keyed on context seen in training. Rotary models stop at the trained number of steps (F4, F12), which suggests the step count is keyed to position. Composition failures name the memorised successor rather than the one in the plan (F17, F19). The data changes that moved each failure (more lengths, more partners per stage) are consistent with this reading, but the experiments do not isolate a mechanism.

## 4. Discussion
The results support three practical rules for small models trained from scratch. (1) Write each step's inputs and state, not the history, and keep operands as single symbols. (2) Train on the widest length range the budget allows, and if the model must extrapolate in length inside a stream, use no positional encoding with a per-document mask. (3) Give every component of a composition several partners in training. Adding more examples of the same few partners did not help (F26), while more distinct partners did (F19, F29).

The rotary failure is worth stating plainly. Rotary positions gave zero extrapolation in every stream run we tried, while the same architecture without positions did extrapolate. We have not identified a rotary variant that does so. We also note that the no-position result is one run.

The composition results are narrower than the headline. Diversity fixed selection for the internal language and for English, but two of four internal held-out pairs stayed unstable across seeds (F19), and the three-operand copy stays hard. A chain of arbitrary length has not been tested (F22).

## 5. Limitations
- **Seeds.** Most results are single runs. Some differences are within seed spread, for example addition extrapolation across seeds (0.86, 0.37, 0.80; F8). The project plan asks for at least three seeds per reported number. Many findings do not meet that.
- **Training budget.** No learning-rate sweep was run for most conditions, and token budgets were fixed (100M to 400M) rather than scaled with model size. The size ladder is a smoke test.
- **Synthetic data.** The reasoning tasks use small vocabularies, and the English requests are templates and generator-written phrasings, not human-written English. Held-out phrasings and held-out pairs were chosen by hand in several experiments.
- **Evaluator defects.** The English agent figure in one finding was an evaluator error and was corrected in a later finding. A decoder defect that scored a trained turn as 0.16 was re-measured at 0.985. These corrections are reported in the source logs.
- **Table values.** One headline range in the source (0.95 to 1.00 at trained length for held-out chains, F22) does not match its own table, whose first column contains 0.89. Table values are used here.
- **Scope.** Length extrapolation was tested to two lengths beyond training. Multi-turn and cross-register behaviour was not tested.

## 6. Conclusion
In small from-scratch transformers, length extrapolation and composition depend less on model size than on two properties of the training data: what the written step contains, and how many contexts each procedure appears in. Rotary positions produced no length extrapolation in our stream runs, while a no-position model with per-document attention reached 0.74 at 6 digits. Composition of unseen pairs failed with few partners per stage and succeeded with many, reaching 0.81 in English for held-out pairs. These are measured effects from one-run experiments with the limits above, and the next step is multi-seed replication of the no-position result and the English composition result.

## References
- Coconut, "Training Large Language Models to Reason in a Continuous Latent Space" (2024). https://arxiv.org/abs/2412.06769. Named in the findings (F5, F6).
- Zhu et al. (2025), "Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought." https://arxiv.org/abs/2505.12514. Named in F5.
- SIM-CoT, latent instability and per-step supervision (2025). https://arxiv.org/abs/2509.20317. Named in F6.
- Fan et al. (2025), named in F3 and F4 for the no-position result. The reference list in the project docs gives "Looped Transformers for Length Generalization" (ICLR 2025), https://arxiv.org/abs/2409.15647; the docs do not list the author names for this entry.
- Deng et al. (2023) and Deng et al. (2024), named in F3 and F5 on depth and multiplication. The docs give no full bibliographic entry.
- TinyStories, https://arxiv.org/abs/2305.07759. Named as the prose corpus in `docs/references.md`; author names are not given in the docs.
