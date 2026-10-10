# Query Restatement Fixes the Decisive Label but Not Multi-Step Proof Validity: A Small-Scale Study of Synthetic Signed-Rule Deduction

Eric Lam (independent researcher, Toronto)

## Abstract

We ask whether a small decoder-only Transformer, trained from scratch on synthetic signed-rule deduction, can write proofs that pass a strict checker, rather than only producing correct final answers. Across seven 480k-to-1M-row training runs and one stopped run, we vary proof style, decoy rules, query restatement, input format and data volume. A checker-verified restated query makes the decisive label learnable under teacher forcing, but it does not produce reliable proofs. In the best sample, 48 of 150 outputs were valid proofs, and no sample contained more than two valid depth-3 proofs. Correct final answers outnumber valid proofs by a wide margin in every run. The two data changes we tested, decoy rules and closure-style gold proofs, both reduced valid proofs relative to the baseline ablation. These are single-seed results on one generator, and we report them as a diagnostic rather than a capability claim.

## 1. Introduction

Final-answer accuracy is a weak measure of reasoning. A model can output a correct label while its stated derivation is wrong, and a shortcut that ignores rule prerequisites can score well on final answers alone. We therefore score each output in two parts: the decision, and whether the written proof passes a strict replay checker. A correct answer with an invalid proof is a separate outcome.

The study asks three questions. First, can a small model trained from scratch produce valid multi-step proofs at depths 2 to 4? Second, which training choices change proof validity? Third, do larger data volume and a larger model change the result? The pre-registered contracts for each run defined the bars in advance. None of the runs met its bar.

## 2. Related work

The project's roadmap names several precedents that motivated the setup. Token Assorted combines textual and learned discrete reasoning tokens [R1]. Chain of Draft motivates a concise-language baseline [R2]. Shorthand for Thought learns cross-word supertokens [R3]. Compositional meta-learning shows that training organization can help some generalization tasks while failing others [R4]. The roadmap also cites work showing that intermediate tokens can supply computation even when their content is uninformative [R5], and ProofWriter as a public reference for deduction and proof evaluation [R6]. The scale contracts cite from-scratch symbolic deduction results at 1–3M parameters with 150k–300k examples (Gontier et al. 2020; Brinkmann et al. 2024). The diagnosis report cites scheduled sampling as a possible mechanism for error accumulation during free-running generation (arXiv:1506.03099). The source folder does not include a references file, so the full bibliographic details of these works are not reproduced here.

## 3. Task and checker

Worlds are finite and grounded. They contain 64 proposition names, signed facts and conjunctive rules, with no variables. A literal is either positive or negative, and the explicit opposite of a literal is its sign flip. The answer is ENTAILED if the query is in the least fixed point of the rules, REFUTED if its explicit opposite is, and UNKNOWN otherwise. Worlds whose closure contains both signs of an atom are rejected. Minimum proof depth is proof height: facts have depth 0, and a derived literal has the minimum over applicable rules of one plus the maximum depth of its premises. Depth is computed by the oracle, not taken from the planted chain length.

Decisive answers are balanced across depths 0–4. UNKNOWN examples come from four mechanisms: an isolated atom, an unsupported cycle, a missing conjunctive premise and an opposite prerequisite. A strict checker replays each generated trace against the problem. For UNKNOWN, the output must be a closed certificate over the reachable set. A trace is valid only if it passes replay. The decision is scored independently.

## 4. Experimental setup

Corpora were generated with fixed seeds and split by structural family, so renamed copies of a world stay in one partition. The 1M-row corpora have 696,333, 702,658 and 696,269 training rows (v2, v3 and v4). Ablation corpora A–G each contain 480,000 problems. Corpora A–D were generated but never trained. Validation samples are fixed at 150 rows, balanced at 50 per answer.

The first model has 1,875,072 parameters (4 layers, width 192, 3 heads). The later models have 10,834,944 parameters (6 layers, width 384, 6 heads, feed-forward width 1,536). Models are trained from scratch with AdamW and a loss on target tokens only, in BF16. The tokenizer is fitted on 10,002 training rows. The 1.9M model made one pass over its 696,333 training rows in 43,521 steps. The 10.8M runs used two epochs with cosine decay and batches capped at 32,768 padded tokens. A 57M model was started and stopped at step 7,000 of 43,522 after its label probe showed no sensitivity to either sign flip.

Four conditions structure the comparison. The debug run compared four target formats on 12 problems. The scale runs (SCALE-01 to SCALE-04) use the compact format only. The ablations E, F and G share a seed, configuration and generator family, and differ as follows. E uses the v1 generator with minimal gold proofs. F uses the v2 generator with decoys and closure-style gold proofs. G uses the v2 generator with decoys and minimal gold proofs. All three restate the query in the trace before the decision. For a shared comparison, E, F and G, along with SCALE-01a and SCALE-03, were also scored on the same 150-row sample drawn from the v2 corpus.

## 5. Results

Table 1 summarizes the runs. Own-sample numbers come from each run's own validation rows; shared-sample numbers use the v2 sample.

**Table 1.** Results by run. Valid counts are proofs that pass the checker.

| Run | Data | Own-sample correct / valid (of 150) | Shared v2 sample correct / valid | Notes |
|---|---|---|---|---|
| Debug, 4 formats | 12 problems | decision-only 0/12 correct; proofs 0/12 valid | — | Memorized 12/12 training traces |
| 1.9M, one pass | v2 | 58 / 6 | — | Depths 2–4: 0 valid |
| SCALE-01a, 10.8M | v2, 2 epochs | 75 / 48 | 75 / 48 (reference) | Depth 3: 2/18; depth 4: 1/13 |
| E, 10.8M | 480k, v1, restated | 90 / 25 | 102 / 33 | Given gold proof, label 77/100 |
| F, 10.8M | 480k, v2 decoys, closure, restated | 60 / 10 | 68 / 20 | Given gold proof, label 52/100 |
| G, 10.8M | 480k, v2 decoys, minimal, restated | 69 / 12 | 62 / 14 | Given gold proof, label 91/100 |
| SCALE-03, 10.8M | 1M v3, restated | 98 / 33 | 112 / 45 | Depths 3–4: 0 valid |
| SCALE-04, 10.8M | 1M v4, structured input | 107 / 41 | — | Depth 3: 0/23; depth 4: 1/20 |

Three results stand out. First, the restated query moves the label. Given the gold proof, the decisive label was at chance for SCALE-01a (24/50 ENTAILED and 24/50 REFUTED). With the restatement, it was correct on 77/100 (E), 91/100 (G) and 99/100 (SCALE-03 and SCALE-04). Second, proof validity stays low at every scale. Across all runs, the largest number of valid depth-3 proofs in any sample was 2 of 18, and depth 4 reached one valid proof in two samples. Third, correct answers vastly outnumber valid proofs. SCALE-04 reached 107 correct answers but 41 valid proofs, so 66 correct answers had invalid evidence. The most frequent first checker error was a premise-reference mismatch: 92 of 150 outputs for SCALE-04 and 99 for SCALE-03.

The ablations do not favour the proposed data changes. On both the own and the shared samples, F and G produced fewer valid proofs than E (E: 25 and 33; F: 10 and 20; G: 12 and 14). The per-sample comparison of F and G is mixed: F was lower on its own sample and higher on the shared sample. Sensitivity probes show a one-sided pattern. In E, F, G and SCALE-03, the label changed under query-polarity flips and never under derived-sign flips. In SCALE-04 the pattern reversed: the label never changed under query-polarity flips and changed in all 77 derived-sign flips. No run was sensitive to both signs, which the XOR structure of the label requires. We do not interpret this reversal.

Scale did not clearly help. On the shared sample, SCALE-03 scored 45 valid proofs against 48 for SCALE-01a. SCALE-04 reached 41 valid proofs on its own sample, with no valid depth-3 proof. The bars set before the runs (more than 48 valid on the shared sample, and valid proofs at depths 3 and 4 for SCALE-04) were not met. Neither was the 80% validity bar for SCALE-01.

## 6. Discussion

The restated query is a useful, checkable intervention. It makes a decisive label learnable under teacher forcing. But the label is not the bottleneck. The failures concentrate in premise selection and reference consistency. A model that writes a syntactically valid trace can still cite the wrong premises, add extra steps, or stop at the wrong point. In the 1.9M diagnostic, correct preceding proof tokens raised final-answer accuracy from 38.7% to 64.7%, which suggests that error accumulation during free-running generation is one contributor and not the only one.

The result is consistent with a simple reading: a small model trained by next-token prediction on this corpus learns the surface format and some answer structure before it learns the selection operations that make a proof valid. The data changes meant to reduce shortcuts (decoys and closure-style gold) did not help at 480k rows. A reasonable next step, which the project's own notes recommend, is to isolate fact retrieval, one-rule application and reference matching in a small varied curriculum, then track free-running validity during training. That step has not been run.

## 7. Limitations

- The corpora come from one synthetic grounded-rule generator with 64 proposition names and depths up to 4. The results say nothing about natural language, variable binding or transfer.
- Each configuration has a single seed. Validation samples contain 150 rows. Differences of a few rows, such as 45 against 48, are too small to separate reliably, and we report no confidence intervals.
- Checkpoint selection in the scale runs used a probe drawn from the same validation sample, so validation numbers may be optimistic. Test partitions were not evaluated.
- The comparisons are not fully controlled. Corpora A–D were generated but never trained, so the planned 2×2 ablation is incomplete. SCALE-01a, SCALE-03 and SCALE-04 change corpus, input and tokenizer across steps.
- The cross-evaluation of ablation E on corpus D has no output file. The 57M run stopped early with no summary.
- Timings and memory figures come from single runs. The sensitivity probes are teacher-forced and one-sided, and the reversal in SCALE-04 is unexplained.

## 8. Conclusion

In this bounded setting, a checker-verified restated query makes the decisive label learnable, but a 1.9M to 10.8M parameter Transformer trained from scratch on up to one million synthetic problems did not produce reliable multi-step proofs. The best validation sample reached 48 valid proofs of 150, with no reliable valid proofs at depths 3 and 4. Decoy rules and closure-style gold proofs reduced validity relative to the baseline ablation. The study is a negative diagnostic: it shows where the failures lie and which bars were not met, not that such models cannot learn deduction.

## References

Entries are limited to works named in the project's source documents. The source folder contains no references file, so full bibliographic details are not reproduced.

- [R1] Token Assorted, as named in ROADMAP.md.
- [R2] Chain of Draft, as named in ROADMAP.md.
- [R3] Shorthand for Thought, as named in ROADMAP.md.
- [R4] Compositional meta-learning, as named in ROADMAP.md.
- [R5] Work on intermediate tokens supplying computation, as described in ROADMAP.md.
- [R6] ProofWriter, as named in ROADMAP.md.
- Gontier et al. 2020; Brinkmann et al. 2024. Cited in SCALE_04_SPEC.md.
- Scheduled sampling, arXiv:1506.03099. Linked in reports/gpu_million_diagnosis_20260911.md.
