# Sleep on Self-Generated Dreams and Running-Sum Links for Continual Learning: Controlled Results and Negative Findings

Eric Lam (independent researcher, Toronto)

## Abstract

We study whether a small learner can take in new material one slice at a time without replaying stored data. We test three mechanisms. First, a byte-level GPT trained on six child-directed and general English domains, one per day, is rehearsed each night on its own dreams, with a term that asks the model to keep its previous predictions on those dreams. Across three seeds, the best of four sleep settings reduces average forgetting from 0.58 to 0.08 bits per byte, but it learns each new domain slightly less well and misses its predicted final average. A single-seed, pre-registered version of the same idea reduces forgetting from 0.583 to 0.205 with essentially the same learning (1.828 against 1.825 bits per byte). Second, a running-sum link (W = A(B + λI)⁻¹, updated by adding to A and B) on frozen features keeps day-one classes on 20-day streams of CIFAR-100 image kinds (0.79 at day 20) and 20 Newsgroups topics (0.62), where a normal two-layer network falls to 0.00. The link matches an all-at-once fit, but it breaks when the features under it are allowed to learn. Third, practice alone did not improve results, and several earlier results were withdrawn once privileged information was removed. We report these as negatives. The main finding is that sleep-style rehearsal reduces forgetting in a learning network but does not remove its cause.

## 1. Introduction

A learner that is updated on a stream forgets what it learned earlier. In a network whose weights are shared by everything it knows, a gradient step for new material moves weights that old material depends on. Replay of stored data is the standard remedy, but it needs storage and a copy of the past. Complementary-learning-systems accounts suggest a different route: a fast store keeps episodes, and a slow cortex learns from interleaved replay, so that new learning does not wipe old knowledge (McClelland, McNaughton and O'Reilly, 1995). Later work points out that replay can be weighted by importance rather than applied uniformly (Kumaran, Hassabis and McClelland, 2016), and that replay can be prioritised by expected gain (Mattar and Daw, 2018). Schemas allow one-trial learning once a shared structure exists (Tse et al., 2007). On the theory side, fixed-size zero forgetting on arbitrary task sequences is reported as impossible with shared non-linear features (Peng and Risteski).

This paper asks a narrow question. Can a learner keep old material without stored data, using only (a) rehearsal of its own dreams and (b) a running-sum read-out on fixed features? We ran controlled comparisons on byte-level and word-level models and on frozen image and text features. We report the results that were pre-registered, the results found after the grid was run, and the failures.

## 2. Method

### 2.1 Continual GPT and sleep on dreams

The model is nanoGPT (Karpathy, github.com/karpathy/nanoGPT) redesigned for this test. It is byte-level (256 symbols), with 4 blocks, width 256, 4 heads and a 256-byte context. It is trained from scratch. The data are six domains of BabyLM 10M "clean" text: CHILDES speech to children, Simple Wikipedia, BNC spoken, Gutenberg, QED subtitles and children's stories. Each domain is one "day". A day is 1,500 steps of 32 × 256 bytes. Each day loads the previous checkpoint, learns one domain, saves, and is then tested on every domain's held-out text (the last 5% of each file).

Sleep on dreams works as follows. Before a night, a frozen copy of the model writes 512 dreams of 256 bytes, each started from a newline. During the night the model learns the day's text and, in addition, a KL term asks it to keep its predictions on those dreams close to the frozen copy's. No old text is stored. Two knobs were varied: the KL weight (1 or 3) and the number of dreams (512 or 2,048).

### 2.2 Running-sum links on frozen features

A link maps a feature vector x to a name vector y. It keeps two running sums, A += y xᵀ and B += x xᵀ, and reads out W = A(B + λI)⁻¹. The features come from frozen encoders: a pretrained ResNet-18 for CIFAR-100 images, and a fixed word-hashing map (4,096 numbers, word counts squashed) for 20 Newsgroups posts. Each kind or topic has a random name vector. The answer is the nearest name among those already heard. Image streams add 5 new kinds per day over 20 days; text streams add 1 new topic per day over 20 days. The comparison is a two-layer scikit-learn network (Adam) trained day by day, and the same network trained on all days mixed.

### 2.3 Predictions

Predictions were written before the runs and kept in the scripts. For the sleep-on-dreams GPT, G1 predicted that plain would forget at least 0.10 bits per byte on average; G2 that the combined slots-plus-dreams arm would forget at most a third as much as plain; and D1 that the best sleep setting would reach a final average of at most 1.92 across the six domains. For the link, the three predictions were: day-1 classes at day 20 within 3 points of an all-at-once link; a normal network's day-1 classes below 5% at day 20; and overall accuracy at day 20 of at least 50% on CIFAR-100 and at least 60% on 20 Newsgroups.

## 3. Experiments and results

### 3.1 Sleep on dreams: one seed, pre-registered

| Arm (cgpt.py, one seed) | Just learned (bits/byte) | Forgetting | Final average |
|---|---|---|---|
| Plain (day by day) | 1.825 | 0.583 | 2.311 |
| Slots | 2.293 | 0.244 | 2.497 |
| Dreams (plain + sleep on dreams) | 1.828 | 0.205 | 1.999 |
| Slots + dreams | 2.334 | −0.003 | 2.331 |
| Joint (all six domains mixed) | — | — | 1.812 |

Dreams alone learned as well as plain and forgot about 35% as much. The slot cortex (1,024 key/value slots, 32 read per position) learned worse from day 1, so its low forgetting partly reflects learning less. G1 passed (plain forgot 0.583 against a line of 0.10). G2 passed for the combined arm (−0.003 against a line of 0.19). G3 (learning within 0.05 of plain) and G4 (final average below plain's and within 0.10 of joint's) failed.

### 3.2 Sleep strength: three seeds, after the grid ran

| Arm (cgpt_gap.py, 3 seeds) | Just learned | Forgetting | Final average |
|---|---|---|---|
| Weight 1, 512 dreams (rerun) | 1.833 | 0.204 | 2.003 |
| Weight 3 | 1.840 | 0.137 | 1.954 |
| 2,048 dreams | 1.831 | 0.168 | 1.971 |
| Weight 3, 2,048 dreams | 1.854 | 0.083 | 1.923 |

Reference values from the same protocol: plain 0.58 forgetting and 2.31 final; joint ceiling 1.81. D1 failed by a hair (1.923 against a line of 1.92). D2 passed (learning within 0.03 of plain). D3 failed (weight 3 alone forgot 0.137 against a line of 0.12). The authors' post hoc summary is that the two knobs add up: stronger keeping combined with wider dreams cut forgetting to 0.08, about 14% of plain's, at a learning cost of about 0.02 bits per byte. We note that this best arm was chosen from a four-arm grid after the first run, so the 0.083 figure is a post hoc result on three seeds, not a pre-registered one.

### 3.3 Running-sum links on 20-day streams

| Stream (twenty_days.py) | Measure at day 20 | Link | Normal network | Control |
|---|---|---|---|---|
| CIFAR-100, 5 kinds per day | Day-1 kinds | 0.79 | 0.00 | — |
| CIFAR-100 | All kinds so far | 0.60 | 0.05 | All at once: link 0.60, network 0.58 |
| 20 Newsgroups, 1 topic per day | Day-1 topics | 0.62 | 0.00 | — |
| 20 Newsgroups | All topics so far | 0.60 | 0.05 | All at once: link 0.60, network 0.54 |

The link learned day by day gives the same numbers as the link trained on all days at once, up to float32 rounding. Predictions 2 (the normal network's day-1 classes below 5%) and 3 (overall accuracy at least 50% on CIFAR-100 and 60% on 20 Newsgroups, within 10 points of the all-at-once network) passed on both datasets. Prediction 1 passed on 20 Newsgroups (0.62 against 0.60). For CIFAR-100 the file also records PASS, but the day-1 score (0.79) is not within 3 points of the all-at-once figure (0.60), so the wording of this prediction is ambiguous.

The decline of day-1 scores over days in the link tracks the decline of the all-kinds score. The authors attribute this to more competitors, not to forgetting.

### 3.4 Negatives and withdrawn results

- Practice did not help. A closed-loop practice run on unseen List Functions moved accuracy from 45.0 to 45.7, which the authors call noise.
- The link breaks when the features beneath it learn. In a falsification test, day-1 digits fell from 1.00 to 0.00 when the area below learned.
- Night length matters for one-sentence learning. With dreams, a 10-step night cost +0.05 bits per byte on ordinary text, a 30-step night +0.03, and a 100-step night +0.003 to +0.010. Without dreams, 100 steps cost +0.89 to +1.03. The gentlest, longest night was the cheapest.
- In the 13-day Wikipedia stream on a pretrained 39M model, pages were learned the morning after their night at 7–10 out of 10 probes. By day 13, pages from days 1–5 had fallen to 1–5 out of 10, and general English moved from 6.34 to 6.85 bits per word piece. No plain fine-tuning baseline was run for this stream.
- A one-shot memory for new words, a fact-sorting rule and a fact layer all looked good until privileged information (answer boundaries, fact identities) was removed. They were withdrawn.

## 4. Discussion

The results separate two effects. Rehearsing a model's own dreams, with a term that keeps its old predictions, reduces forgetting in a learning network by a large factor. In the pre-registered run it did so with essentially no loss of learning. But the effect is partial. The strongest arm still leaves a gap to the all-at-once ceiling, and older material fades over a long stream. The dreams protect what they happen to show. The effect on one-sentence facts is also uneven: a dream start at a newline let one sentence take over about 500 of 512 dreams, and varied starts fixed that, while leaving a modest, steady share of dreams for the fact.

The link result is clean and is the strongest positive in this study. On frozen features, a running-sum read-out is exact and order-free, and it keeps day-1 classes where a normal network forgets them completely. The result does not extend to learned features. In the falsification run, the link fell apart when the area beneath it learned.

Practice added little, and the failures were informative. Gains came from structure chosen by hand, and several results depended on information the model could not have at runtime. Once those were removed, the striking results disappeared. For this reason the authors judged the findings too shallow. Each experiment reduced forgetting, but none removed the cause. A tentative open question follows: is there a learning rule or architecture in which deep, feature-learning networks keep the property that sums do not overwrite each other?

## 5. Limitations

- Most results are one seed. The key ones are three seeds. The models are small: a 3M–39M range for most runs, the 39M pretrained gpt2-mini for the runtime stream, and an 85M GPT that was stopped at day 3.
- The 0.083 forgetting figure comes from the best of four sleep settings, chosen after the grid ran. The pre-registered single-seed figure is 0.205.
- The 13-day stream used 125 verbatim probes written by one tester and has no plain fine-tuning baseline.
- Dream-based GPT arms learned slightly less well than plain (1.854 against 1.825–1.834).
- The link results use frozen encoders and one stream order per dataset. The CIFAR-100 first prediction is ambiguous as written (see Section 3.3).
- The "all at once" references are a single two-layer network, not a tuned best-possible learner.
- The pulse-code and word-meaning results were not carried into this paper's main claims.

## 6. Conclusion

Sleep on a model's own dreams, with a term that keeps its old predictions, reduces forgetting in a byte-level GPT on six domains: from 0.583 to 0.205 bits per byte in a pre-registered single-seed run, and to 0.083 in the best of four settings on three seeds. The stronger setting costs a little learning (1.854 against 1.825), the pre-registered setting costs almost none, and neither removes forgetting. A running-sum read-out on frozen features keeps old classes on 20-day streams of CIFAR-100 and 20 Newsgroups, where a normal network does not, and matches an all-at-once fit. Once the features beneath the read-out learn, the advantage breaks. Practice and structure added without new kinds of pieces did not produce gains. We report these as small, partly negative results.

## References

- Karpathy, A. nanoGPT. github.com/karpathy/nanoGPT (the model code this project redesigned; `model.py`).
- BabyLM 10M "clean" corpus (cambridge-climb/BabyLM), as used in the six-domain protocol.
- McClelland, J. L., McNaughton, B. L., and O'Reilly, R. C. (1995). Complementary learning systems and the hippocampus and neocortex (PubMed 7624455).
- Kumaran, D., Hassabis, D., and McClelland, J. L. (2016). Schemas, interleaved replay and importance-weighted replay (PubMed 27315762). Listed as not re-checked in the project's notes.
- Tse, D. et al. (2007). Schemas allow one-trial learning. Science 316:76–82.
- Mattar, M. G., and Daw, N. D. (2018). Replay is prioritised by expected gain (Nature Neuroscience, s41593-018-0232-z). Listed as not re-checked in the project's notes.
- Peng and Risteski, on fixed-size zero-forgetting learning with shared non-linear features. Cited in the project's maths notes without year or venue.
- CIFAR-100 (100 object kinds), with a frozen ResNet-18 as the vision feature extractor. No citation details are given in the project files.
- 20 Newsgroups (20 forum topics). No citation details are given in the project files.
- gpt2-mini (39M word-piece model, huggingface.co/erwanf/gpt2-mini), used for the runtime stream.
