# Fact Memory Slots on a Frozen Transformer: Exact Addressing, Single-Pass Composition, and Bounded Retraction

Eric Lam (independent researcher, Toronto)

## Abstract
A frozen transformer cannot learn new facts, and a memory layer that stores them is only useful if the stored facts can be found, composed, and removed. We test a slot memory on a frozen 3.21M-parameter backbone trained on a synthetic entity-relation universe. Writing facts into slots with a learned, hidden-state address separates poorly: 98.70% of kept facts collided with written slots, and kept-fact accuracy fell from 0.996 to 0.243. Addressing slots by a hash of the content tokens removes collisions, giving online recall of 1.000 for 200 new facts and kept accuracy of 1.000 on 1,800 old facts. Composition over slot-stored facts does not occur inside a single forward pass (0.010 on 3,402 two-hop consequences). A symbolic loop that asks one hop at a time over the same memory scores 1.000, but it is handed the decomposition. Zeroing a fact's slot removes it, and withdrawing a fact's recorded derivations removes 0.995 of its consequences with 0.000 collateral in 24 retractions. These are single-run results on a closed synthetic universe.

## 1 Introduction
Knowledge editing methods can recall an injected fact but often fail on multi-hop questions whose answers depend on it. The reported band for MQuAKE-CF is about 15% to 34% (Vicuna-7B, best cited 33.8%). This project asks whether a memory written at inference time can avoid that failure while keeping the backbone frozen and keeping retraction exact. The project's hypothesis is stated as gates: G2 requires high recall of new facts with minimal loss on old ones, and G3 requires multi-hop accuracy over facts learned online that substantially exceeds the band.

We report what passed and what did not. The main negative results are that hidden-state addressing fails, that a single forward pass does not compose over slot facts, and that exact-looking scans inherit a confabulation rate.

## 2 Method
**Backbone.** A nanoGPT fork with 4 layers, 4 heads, and 256 embedding dimensions (3.21M parameters). It is trained from scratch on a universe of 200 nonsense-word entities. Relations are composed as nested "of" phrases, and each entity has a fixed out-degree, so the ratio of inferred to atomic facts is a settable parameter (φ = 10 in the reported run). Training runs without early stopping. Held-out composition rose from 0.061 at 2,500 steps to 0.918 at 100,000 steps, which is a grokking transition similar in shape to Wang et al. (NeurIPS 2024).

**Memory.** A key-value slot table is read in the forward pass. Two address schemes were tested. In the hidden-state scheme, the address is a learned projection of the hidden state. In the token scheme, the address is a rolling hash of the content tokens of (relation, subject), emitted only at the answer position. A slot read with all slots zeroed gives the ablation.

**Write protocol.** Phase 1 trains the full model on 1,800 atomic facts and their compositions, holding out 200 atomic facts. Phase 2 freezes the backbone and trains only the slot table on the 200 held-out facts (1,048,576 trainable parameters in the 2,048-slot runs). Writes can be unconfined (gradient over the full table) or confined to fresh rows.

**Language and chaining.** A 0.89M-parameter parser maps free English wording to the canonical token line that serves as the slot address. Its head points at input positions rather than generating words, because a generative head scored 0.000 on held-out content words. Two-hop questions are answered one hop at a time, with each answer used as the next address. A board outside the slots records the premises of each derived conclusion.

**Evaluation.** The main set has 3,402 two-hop consequences of slot-written facts. Retraction is measured on 24 premises, each with 200 control facts. Provenance is checked against the ground-truth entailment graph.

## 3 Experiments and Results

### 3.1 Storage and addressing
| Condition | Online recall | Kept atomic | Note |
|---|---|---|---|
| Hidden-state address, unconfined write, 20,000 steps | 1.000 | 0.243 | Memory off: kept 0.996 |
| Hidden-state address, confined write | 0.665 | 0.234 | Confinement does not fix interference |
| Token-hash address, 20,000 steps | 1.000 | 1.000 | 0 of 1,800 kept facts colliding |

Under hidden-state addressing, 1,776 of 1,800 kept facts collided with written slots (98.70%). The cosine margin between an online query and its nearest intruder was 0.0351 (0.8836 vs. 0.8486). Three read schemes on the hidden-state address did not fix the damage: softmax (kept 0.243), gated (0.202), and threshold (0.091).

### 3.2 Composition
| Condition | Result | Control |
|---|---|---|
| Backbone, grokked, held-out composition | 0.918 | Chance 0.44% |
| Slot-stored facts, token address, 20,000 steps, online composition | 0.009 | Backbone 0.918 |
| Single forward pass, 3,402 two-hop consequences | 0.010 | n/a |
| Symbolic two-hop chain over the same memory, decomposition supplied | 1.000 | Single pass 0.010 |

The composition gap is structural rather than a tuning failure: the second address is a function of the first answer, which a single read keyed on surface tokens cannot express. The chaining result shows that the memory can answer two-hop questions when the decomposition is provided. It does not show that the transformer composes.

### 3.3 Free-form input
Using the parser on the same 3,402 questions, with a distractor name planted in each sentence, gives a correct canonical read of 1.000 and a correct answer of 1.000. On held-out content words, the generative head scored 0.000 and the pointer head scored 0.438 (1-hop). On held-out wordings, the pointer head scored 0.895 (1-hop) and 0.887 (2-hop), against 0.600 and 0.762 for the generative head. Sweeps over parameters (0.14M to 14.88M) and steps (4,000 to 100,000) did not move held-out wording accuracy. Sweeping wording diversity from 4 to 32 moved it from 0.445 to 0.895. The parser's invariance over 80 phrasings was 1.000, but this covers only 40 project-written wordings.

### 3.4 Retraction
| Measure | Result | Control |
|---|---|---|
| Premise recalled after its slot is zeroed | 0.000 | Backbone: no retraction |
| Consequences withdrawn (G5a) | 0.995 | 3,402 consequences |
| Collateral damage to unrelated consequences | 0.000 | 200 controls per retraction |
| Recorded premises match entailment graph | 3,402 of 3,402 | n/a |
| Consequences keeping the same answer word via a different intermediate | 0.005 | Old derivation gone in all cases |

Retraction here is a set operation on the board, and does not require gradient updates. A zeroed slot means the premise cannot be recalled from the substrate.

### 3.5 Query operations
Exact set or value match: `follow` 1.000, `same` 1.000, `none` (refusal) 1.000, `reverse` 0.970, `count` 0.970. `reverse` scans all entities. Before a confidence floor, it gave 0.401 invented names per true edge and exact-match 0.660. A floor at 0.99 cut invented names to 0.032 and raised exact match to 0.970. The floor is below all true edges (minimum 0.998), so the separation is empirical rather than structural.

## 4 Discussion
The results support a narrow claim. Facts can be written into a frozen model's memory at inference without damaging old facts, but only when the address is exact. The hidden-state address failed for a reason the project traces to geometry: hidden states of different queries sit close together (mean cosine 0.8836 to the nearest online query, against 0.8486 for an intruder), and no read scheme separated them.

Two results are about the boundary between the network and the surrounding system. The first is that composition inside one pass did not appear for slot facts, while the same memory supports two-hop answers when the question is split. The second is that an operation which reads memory N times inherits N chances to confabulate. `reverse` shows this. Filtering reduced the rate but did not remove it. The project's conclusion is that abstention, not a better filter, is the remaining route to exact scans. Abstention is not built.

Retraction is the cleanest result, but its meaning depends on the substrate. In this design, deletion is an address operation for stored facts and a bookkeeping operation for derived conclusions. The result does not show that a learned network can forget exactly.

## 5 Limitations
- Scale: 200 nonsense-word entities, a closed vocabulary, and paths of depth two in the online experiments.
- Seeds: most results are single runs. The Llama reference harness pooled seeds 0-3 at small n (176, 144, and 96 questions).
- Language: the parser was trained on 40 project-written wordings. Real English was not tested.
- Baseline: the Llama-3-8B reference injects facts in context. Its multi-hop numbers match the magnitude of the MQuAKE-CF band, but the method differs from weight editing.
- Gates not measured: adversarial recovery (G5b) and competition margin (G5c) on the slot system; abstention (G4); arithmetic (G6); decision-making (G7).
- Chaining: the two-hop 1.000 uses a decomposition supplied by the harness.
- Persistence: the board is per session, and rules are not implemented.

## 6 Conclusion
A slot memory on a frozen backbone can store and recall facts written at inference when the address is an exact function of the content tokens. Hidden-state addressing fails. Single-pass composition over slot facts does not occur (0.010), while a symbolic loop over the same memory reaches 1.000 when handed the decomposition. Retraction by zeroing a slot and removing its derivations is exact in the tested universe, and scans need abstention before they can be trusted. The next steps in the project are abstention, arithmetic with provenance, and rules stored as facts.

## References
- Vinyals et al., 2015. *Pointer Networks*. arXiv:1506.03134.
- Wang et al., NeurIPS 2024. arXiv:2405.15071. (Grokking and multi-hop composition; named in the project notes with its training setup.)
- Hoffmann et al., 2022. arXiv:2203.15556. (Compute-optimal training; named as "for every doubling of model size the number of training tokens should also be doubled.")
- Kaplan et al., 2020. arXiv:2001.08361.
- Geva et al., arXiv:2012.14913. (Named for the feed-forward abstention argument.)
- *Large Memory Layers with Product Keys*, arXiv:1907.05242.
- *Memory Layers at Scale*, arXiv:2412.09764.
- *MQuAKE: Assessing Knowledge Editing via Multi-Hop Questions*, arXiv:2305.14795.
- Doyle, 1979. https://www.sciencedirect.com/science/article/abs/pii/0004370279900080. (Belief repair; the project's contradiction handling follows its move.)
- Karpathy, *nanoGPT*. Reference implementation, cloned locally; no version recorded.
