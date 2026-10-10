# Verified multi-step deduction in small transformers
**Dates:** 2026-09-11 → 2026-09-12 · **Status:** paused (last recorded run finished 2026-09-12; no follow-up contract is written) · **Hardware/scale:** one local NVIDIA RTX 4080 SUPER (PyTorch 2.11.0+cu128); data generated on local CPU; synthetic corpora of 480,000 and 1,000,020 problems; models of 1,875,072 and 10,834,944 parameters, plus a 57M-parameter run that was stopped early

## Question
Can a small decoder-only Transformer trained from scratch write checkable multi-step proofs for synthetic signed-rule deduction, not just correct final answers? Which training changes (proof style, decoy rules, a restated query, structured input, more data) move proof validity?

## Approach
- **Task:** finite, grounded signed-rule worlds with 64 proposition names and no variables. Answers are ENTAILED, REFUTED or UNKNOWN. Decisive answers are balanced across minimum proof depths 0–4. UNKNOWN is certified by a closed reachable set and covers four mechanisms.
- **Checker:** a strict proof checker replays every generated trace. A proof counts as valid only if it passes. The final decision is scored separately, so "correct answer, invalid proof" is its own category.
- **Corpora:** the 1,000,020-problem corpora were split as follows. v2: 696,333 train / 157,684 validation / 146,003 test. v3: 702,658 / 149,188 / 148,174. v4: 696,269 / 151,826 / 151,925. Ablation corpora A–G each contain 480,000 problems. Corpora A–D were generated, but no training run on them is recorded.
- **Validation:** fixed 150-row samples, balanced at 50 per answer. The ablation and scale runs use separate samples. The v2 sample of 150 rows was re-scored for several checkpoints.
- **Models:** a 1,875,072-parameter model (4 layers, width 192, 3 heads) for the first GPU run. A 10,834,944-parameter model (6 layers, width 384, 6 heads, feed-forward width 1,536) for the scale and ablation runs. A 57M-parameter model was started and stopped at step 7,000.
- **Training:** from scratch. AdamW, loss on target tokens only, BF16. The tokenizer was fitted on 10,002 training rows. The 1.9M run made one pass (43,521 steps, batch 16, learning rate 0.0005). The 10.8M runs used two epochs with cosine decay, batches capped at 32,768 padded tokens, and learning rate 0.0005.
- **Conditions compared:** (1) the debug run compared four target formats on 12 problems. (2) The scale runs use the compact format only. (3) The ablations E, F and G differ in generator (v1 or v2 with decoys), gold proof style (minimal or closure), and whether the query is restated in the trace. All three restate the query.

## Results

| Run | Result | Baseline / control | Verdict |
|---|---|---|---|
| Debug, 4 formats, 500 steps (`gpu_debug_20260911`) | Training: 12/12 traces reproduced in all formats. Validation (12 problems): correct decisions 0/12 decision-only, 2/12 compact, 4/12 concise English, 2/12 full English. Valid proofs 0/12 in every proof format. | Same initial weights and the same 12 problems across formats | Memorized the training set; no valid validation proof |
| 1.9M model, one pass over 696,333 problems (`gpu_million_20260911`) | 58/150 correct (38.7%); 6/150 valid proofs (4.0%). Valid by depth: d0 5/22, d1 1/23, d2 0/14, d3 0/21, d4 0/20, UNKNOWN 0/50. | Post-hoc shortcut that treats rule heads as established: 113/150 correct, not a valid reasoner | Fail |
| SCALE-01a, 10.8M, million v2, 2 epochs, 21,762 steps (`gpu_scale_10m_20260911`) | 75/150 correct; 48/150 valid. Valid by depth: d0 14/25, d1 14/25, d2 4/19, d3 2/18, d4 1/13, UNKNOWN 13/50. | 1.9M run: 6/150 valid. Contract bar: 80% valid (120/150) | Partial; below bar |
| SCALE-01b, 57M, same data | Stopped at step 7,000 of 43,522; no summary. Probe at step 6,000: label changed in 0/40 query-polarity flips and 0/31 derived-sign flips. | 10.8M run with same data | Stopped; no result |
| Ablation E, 480k, v1, minimal proof, query restated, 10,484 steps (`gpu_ablation_e_20260911`) | Own sample: 90/150 correct, 25/150 valid. Shared v2 sample: 102/150 correct, 33/150 valid. Label given gold proof: 77/100 decisive. | Compared with F and G (same config, seed 20260932) | Label fixed; proofs still low |
| Ablation F, 480k, v2 decoys, closure proof, query restated, 12,454 steps (`gpu_ablation_f_20260911`) | Own sample: 60/150 correct, 10/150 valid. Shared v2 sample: 68/150 correct, 20/150 valid. Label given gold proof: 52/100. | E | Fewer valid proofs than E on both samples |
| Ablation G, 480k, v2 decoys, minimal proof, query restated, 12,389 steps (`gpu_ablation_g_20260911`) | Own sample: 69/150 correct, 12/150 valid. Shared v2 sample: 62/150 correct, 14/150 valid. Label given gold proof: 91/100. | E | Fewer valid proofs than E on both samples |
| SCALE-03, 10.8M, million v3 with query line, 23,598 steps (`gpu_scale_10m_v3_20260911`) | Own sample: 98/150 correct, 33/150 valid. Shared v2 sample: 112/150 correct, 45/150 valid; valid by depth d0 22/25, d1 16/25, d2 3/19, d3 0/18, d4 0/13, UNKNOWN 4/50. Label given gold proof: 99/100. | SCALE-01a on the same v2 sample: 48 valid | Not above 48; no valid depth-3 or depth-4 proofs |
| SCALE-04, 10.8M, million v4 (structured input, tokenizer v2), 21,917 steps (`gpu_scale_10m_v4_20260912`) | Own sample: 107/150 correct, 41/150 valid. Valid by depth: d0 14/23, d1 10/19, d2 7/15, d3 0/23, d4 1/20, UNKNOWN 9/50. Label given gold proof: 99/100. | SCALE-03 own sample: 33 valid (different sample) | Partial; success bar not met (no valid depth-3 proof) |

Label-probe sensitivity (teacher-forced, decisive rows only): in E the label changed in 78/100 query-polarity flips and 0/85 derived-sign flips; in F, 76/100 and 0/75; in G, 95/100 and 0/75; in SCALE-03, 99/100 and 0/85. In SCALE-04 the pattern reversed: 0/100 query-polarity flips and 77/77 derived-sign flips. No run changed the label under both flips.

## What it showed
- The checker-verified query line made the decisive label learnable under teacher forcing. Without it, SCALE-01a's label was at chance given the gold proof (24/50 ENTAILED, 24/50 REFUTED). With it, the label was right on 77/100 (E), 91/100 (G) and 99/100 (SCALE-03 and SCALE-04). This says nothing about whether the model generates valid proofs.
- Correct final answers were far more common than valid proofs. SCALE-04 had 107/150 correct answers but 41/150 valid proofs. The most frequent first checker error was "Premise references do not match rule" (92 of 150 outputs for SCALE-04, 99 for SCALE-03). Every output parsed and ended with EOS.
- Proof validity did not extend to depths 3–4. Across all scale runs, the most valid depth-3 proofs in any sample was 2 of 18 (SCALE-01a), and depth 4 reached 1 valid proof in two samples (SCALE-01a and SCALE-04).
- The two data changes meant to help did not. On both the own and the shared v2 samples, decoys (G) and closure-style gold (F) produced fewer valid proofs than E (E: 25 and 33; F: 10 and 20; G: 12 and 14). The data ablations are single-seed, so this is a directional result.
- Scale did not raise validity in a clear way. On the shared v2 sample, the 1M-row restated-query model (SCALE-03) scored 45 valid proofs against 48 for SCALE-01a. Structured input with whole-identifier tokens (SCALE-04) reached 41 valid on its own sample and still had no valid depth-3 proof. Neither run met the bar that was written before it ran.

## Limitations
- One synthetic family of grounded signed-rule worlds with depths up to 4. It is not natural language, not a variable-binding task, and not a transfer test. The test partitions were not used for evaluation.
- One seed per run and 150-row validation samples. Differences of a few rows, such as 45 against 48, are too small to separate reliably. No confidence intervals were computed.
- Checkpoint selection in the scale runs used a probe drawn from the same validation sample, so validation numbers may be optimistic (the SCALE-01 contract says so).
- Comparisons are not fully controlled. Corpora A–D were generated but never trained. SCALE-01a, SCALE-03 and SCALE-04 change corpus, input and tokenizer across steps. SCALE-04 deliberately changed two things at once.
- The cross-evaluation of ablation E on corpus D has no output file; its console log records a missing-file error at the time of that run.
- The 1.9M validation diagnostic is clustered across three candidate batches (12, 127 and 11 rows), so it is a diagnostic rather than an unbiased estimate.
- Timings and memory are single-run measurements. Memory is PyTorch peak allocation, not a hardware minimum.
- The sensitivity probes are teacher-forced and one-sided. The source documents do not explain why the reversal in SCALE-04 occurred, and this log does not interpret it.

## Key files
- `README.md`: repository overview, commands and the summary of the first GPU pass. The repository title is "Compact Reasoning LM".
- `ROADMAP.md`: the research program (hypotheses, milestones, and what would count as a result).
- `EXPERIMENT_SPEC.md`: finite signed-rule semantics, minimum proof depth, UNKNOWN certificates and target grammars.
- `MILLION_DATA_SPEC.md`, `STREAM_TRAIN_SPEC.md`: data-scale and streaming training contracts for the 1M corpus and the first pass.
- `SCALE_01_SPEC.md` through `SCALE_04_SPEC.md`, `MILESTONE_02_SPEC.md`: the pre-registered run contracts, bars and caps.
- `DECISIONS.md`, `STATUS.md`: chronological decisions and the status record.
- `src/compact_reasoning/balanced_generator.py`, `varied_generator.py`, `oracle.py`: corpus generation, quotas and gold labels.
- `src/compact_reasoning/checker.py`, `render.py`: strict proof checker and trace renderers.
- `src/compact_reasoning/model.py`, `tokenizer.py`, `streaming.py`, `stream_training.py`, `training.py`, `evaluation.py`: model, tokenizer, streaming loader, training loop and scoring.
- `configs/ablation_data_a.json` through `configs/ablation_data_g.json`: ablation corpus settings.
- `configs/gpu_scale_10m.json`, `configs/gpu_scale_10m_v4.json`, `configs/gpu_scale_57m.json`, `configs/gpu_ablation_10m.json`: run settings.
- `reports/label_probe_20260911.py`, `reports/cross_eval_20260911.py`, `reports/run_ablations_20260911.sh`: label-sensitivity probe, shared-sample evaluation and the ablation runner.
- `reports/gpu_*_20260911*/summary.json` and `validation_predictions.json`: the run results used here.

## Sources
- `README.md`: 1,000,020-problem description; split counts; compact/first GPU result summary.
- `STATUS.md`: best-baseline summary and the 1.9M run figures (58/150, 6/150, 1,875,072 parameters).
- `DECISIONS.md`: SCALE-01a, SCALE-01b (stopped at step 7,000 of 43,522) and SCALE-03 entries. It has no SCALE-04 entry, so the SCALE-04 numbers come from its run directory.
- `reports/milestone_02.md`: debug-run table and figures.
- `reports/gpu_million_20260911.md`: 1.9M run table, depth breakdown and timings.
- `reports/gpu_million_diagnosis_20260911.md`: shortcut diagnostic (113/150) and frozen-weight checks.
- `reports/million_data_20260911.md`: corpus size, split counts and balanced-depth design.
- `SCALE_01_SPEC.md`: contract, caps and success bar (80% valid). Its table lists 10,833,024 parameters; the run summaries record 10,834,944, which this log uses.
- `SCALE_02_SPEC.md`: ablation design, label-amendment figures (24/50, 24/50) and the E/F/G corpus settings.
- `SCALE_03_SPEC.md`: v3 corpus split counts and success bar (more than 48 valid on the v2 sample).
- `SCALE_04_SPEC.md`: v4 corpus split counts and success bar (valid depth-3 and depth-4 proofs).
- `reports/gpu_scale_10m_20260911/summary.json`, `reports/gpu_scale_10m_v3_20260911/summary.json`, `reports/gpu_scale_10m_v4_20260912/summary.json`, `reports/gpu_ablation_{e,f,g}_20260911/summary.json`: step counts, training figures and validation totals.
- `reports/gpu_*/validation_predictions.json`: per-depth counts recomputed from each row's depth, decision and validity fields. The method reproduces the 1.9M run's published depth table exactly.
- `reports/gpu_scale_10m_v3_20260911_external_v2.json`, `reports/gpu_ablation_{e,f,g}_20260911_external_v2.json`: shared v2 sample results.
- `reports/gpu_ablation_{f,g}_20260911_external_d.json`: corpus-D sample results for F and G. No equivalent file exists for E.
- `reports/gpu_*_label_probe.json`: label-given-gold and sensitivity counts.
- `reports/gpu_scale_57m_20260911/STOPPED.json`: stop record for the 57M run.
- `reports/ablations_20260911_console.txt`: missing-file errors during the ablation evaluation.
