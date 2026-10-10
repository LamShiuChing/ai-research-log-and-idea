# Compact continual-learning prototypes (frozen MiniLM + online heads on BANKING77 / CLINC150)
**Dates:** 2026-09-16 (both protocols declared and both experiments completed that day, per the reports; the BANKING77 follow-up diagnostic was declared after the main results, date not recorded) · **Status:** Completed. BANKING77 finished; CLINC150 did not meet its predeclared criterion for an advantage from meta preparation · **Hardware/scale:** Frozen all-MiniLM-L6-v2 encoder (22,713,216 parameters). Text encoding and offline preparation on an RTX 4080 SUPER; deployment timings on CPU with two Torch threads. Python 3.10.11, PyTorch 2.11.0+cu128. BANKING77: 77 intents, 9,992 training-pool and 3,079 test messages. CLINC150: 150 categories, 14,995 train / 2,998 validation / 4,498 test rows.

## Question
Can a small online head on top of a frozen pretrained sentence encoder learn new intent categories from one pass through labeled messages, and does a simple prototype memory do as well or better? A second question is whether preparing the adapter with meta-learning gives a benefit over ordinary supervised preparation and prototype memory on categories withheld from preparation.

## Approach
- **Encoder (both experiments):** frozen `sentence-transformers/all-MiniLM-L6-v2`, FP32, attention-mask mean pooling, L2 normalization, 384 dimensions, maximum 256 tokens. No encoder tuning and no corpus-wide statistics.
- **BANKING77 head:** zero-initialized, bias-free `nn.Linear(384, 77)` (29,568 parameters). One cross-entropy backward pass and one SGD step per example. Labels are per-stream permuted integers; no intent names are given to the model.
- **CLINC150 learner:** residual rank-32 linear adapter (24,576 parameters) followed by normalized features and an expanding cosine classifier (logit scale 10). A new category's row is initialized from its first example's feature (explicit imprinting). One SGD step then updates the adapter and all current rows; the adapter step size is 0.02 times the head step size.
- **Meta preparation (CLINC150 only):** 600 Adam outer updates at 0.001, four tasks per update, five categories per task, four supports and four queries per category, differentiating through online steps (head rate 0.1, adapter rate 0.002). Selected checkpoint: step 200. The ordinary-preparation control uses the same texts and outer steps with a disposable 72-category supervised head.
- **Datasets:** BANKING77 (PolyAI, CC BY 4.0), after duplicate filtering; final support pool of 8,452 training rows, development support 616 and query 924. CLINC150 (authors' `data_full.json` and `domains.json`, CC BY 3.0), out-of-scope rows excluded. Category split: 72 preparation, 24 development, 24 novel categories in familiar domains, and 30 categories in two wholly held-out domains (meta, work).
- **Shots:** BANKING77 at 1, 2, 4, 8 examples per category (interleaved schedule), plus a seven-block new-category schedule of eleven categories per block. CLINC150 at 1, 2, 4 shots on both suites, plus a sequential schedule of six categories per group with four unique examples each.
- **Baselines and controls:** no updates (frozen head); shuffled feedback (same SGD head, random labels); prototype memory (class sums and counts, cosine to means); nearest neighbor (stores every observed embedding); raw linear SGD; unprepared neural; ordinary-prepared neural; prepared prototypes; meta imprint only (no gradient steps).
- **Seeds:** BANKING77 streams 101, 202, 303, 404, 505; selection on development seeds 11, 22, 33. CLINC150 preparation seeds 17, 29, 43, each evaluated on five streams (101, 202, 303, 404, 505); development streams 11, 22, 33.

## Results

| Experiment | Result | Baseline / control | Verdict |
| --- | --- | --- | --- |
| BANKING77 held-out accuracy, online SGD, 1 / 2 / 4 / 8 examples per category | 52.21 ± 1.70 / 60.68 ± 1.60 / 69.77 ± 1.39 / 75.40 ± 0.69 | No updates: 1.29 ± 0.01 at every shot count | Supported: runtime updates generalize to unseen messages |
| BANKING77 paired 8-shot, SGD minus no updates | +74.11 ± 0.69 pp | Frozen head | Supported |
| BANKING77 paired 8-shot, SGD minus shuffled feedback | +73.93 ± 0.74 pp | Shuffled feedback: 1.47 ± 0.44 | Supported: gain depends on correct labels |
| BANKING77 8-shot, SGD minus prototype memory | −4.65 ± 0.86 pp (75.40 vs 80.05 ± 0.92) | Prototype memory | Negative: simple memory is more accurate |
| BANKING77 8-shot, SGD minus nearest neighbor | −5.14 ± 0.92 pp (75.40 vs 80.54 ± 0.61) | Nearest neighbor | Negative |
| BANKING77 1-shot, SGD vs prototype and nearest neighbor | 52.21 ± 1.70 vs 52.24 ± 1.70 (both memories) | Prototype / nearest neighbor | Null: tied at one example per category |
| BANKING77 new-category streams, final accuracy over all 77 categories | SGD 74.19 ± 0.62 | Prototype 80.05 ± 0.92; nearest neighbor 80.54 ± 0.61; no updates 1.29 ± 0.01 | Mixed: learns, but below memory controls |
| BANKING77 first group, full-candidate accuracy, immediately after its block → end of stream | SGD 90.50 ± 3.87 → 65.76 ± 9.93 | Prototype 94.04 ± 2.05 → 80.27 ± 5.51 | Negative: larger drop for SGD |
| BANKING77 follow-up (exploratory): first group restricted to its own labels at end | SGD 90.95 ± 3.88; 92.29 ± 4.18% of final old-group errors predict a later label | Prototype 94.04 ± 2.05 initially, 94.04 ± 2.05 restricted at end; nearest neighbor 94.54 ± 1.22 initially, 94.54 ± 1.22 restricted at end | Exploratory: decline mostly competition with newer labels; oracle restriction, not a deployable score |
| BANKING77 prequential accuracy (predicted before each feedback, includes cold start) | SGD 23.73 ± 0.80 | Prototype 66.53 ± 1.37; nearest neighbor 63.28 ± 2.35 | Negative (reported separately from held-out accuracy) |
| BANKING77 learning-rate selection (development) | Rate 0.1 selected, 73.47 | 0.3: 71.27; 1: 63.58; 3: 59.16; 10: 52.98 | Selection only; rate not changed after test inspection |
| CLINC150 familiar domains, meta-prepared neural, 1 / 2 / 4 shots | 85.52 / 92.04 / 95.56 | Ordinary-prepared neural 86.23 / 91.79 / 95.25; unprepared neural 85.86 / 91.86 / 95.39 | Null |
| CLINC150 held-out domains, meta-prepared neural, 1 / 2 / 4 shots | 71.77 / 77.00 / 84.15 | Ordinary-prepared neural 72.09 / 76.84 / 83.95; raw prototype 72.04 / 78.80 / 85.76 | Null vs ordinary; negative vs raw prototype at 2 and 4 shots |
| CLINC150 primary acquisition, meta minus ordinary-prepared neural | Familiar −0.05 ± 0.90 pp; held-out +0.01 ± 0.58 pp | Ordinary-prepared neural | Null; predeclared criterion failed |
| CLINC150 primary acquisition, meta minus raw prototype | Familiar +0.04 ± 0.49 pp; held-out −1.23 ± 0.26 pp | Raw prototype | Null (familiar); negative (held-out) |
| CLINC150 primary acquisition, meta neural minus its own prepared prototype | Familiar −0.20 ± 0.16 pp; held-out −0.81 ± 0.03 pp | Meta-prepared prototype | Negative: averaging the same features is better |
| CLINC150 meta neural minus imprint-only (no gradient steps) | Familiar +5.48 ± 0.63 pp; held-out +5.89 ± 0.17 pp | Imprint only: 85.56 flat (familiar), 71.75 flat (held-out) | Supported: runtime gradient updates add accuracy beyond first-example imprinting |
| CLINC150 raw linear SGD, 1 / 2 / 4 shots | Familiar 71.19 / 85.89 / 94.22; held-out 60.33 / 72.66 / 82.78 | Meta-prepared neural 85.52 / 92.04 / 95.56; 71.77 / 77.00 / 84.15 | Negative: worse at each shot count |
| CLINC150 sequential, final accuracy over all introduced categories, familiar domains | Meta neural 93.13 | Raw prototype 95.14; ordinary-prepared neural 92.68 | Negative vs raw prototype |
| CLINC150 sequential, final accuracy over all introduced categories, held-out domains | Meta neural 76.19 | Raw prototype 85.76; ordinary-prepared neural 76.94; unprepared neural 78.22 | Negative |
| CLINC150 sequential, held-out domains, first group final accuracy | Meta neural 52.18 (initially 99.37); same group restricted to its labels: 98.22 | Raw prototype 86.55 (initially 98.88) | Negative for full candidate set; restricted result is an oracle diagnostic |

Means pool three preparation seeds and five runtime streams for CLINC150, and five streams for BANKING77. "±" is the sample SD across streams (BANKING77) or across preparation-seed means (CLINC150 paired contrasts). Neither is a confidence interval.

## What it showed
- A frozen pretrained encoder plus a 29,568-parameter head trained with one SGD step per message raised accuracy on unseen BANKING77 messages from 1.29% to 75.40% after eight examples per category. The frozen control stayed at 1.29%, and shuffled feedback stayed near chance (1.47%), so the gain depends on runtime updates with correct labels.
- On the same task, simple memories were more accurate than the SGD head: prototype memory reached 80.05% and nearest neighbor 80.54% at eight shots. The prototype memory has about the same persistent tensor size as the SGD head (118,888 versus 118,272 bytes), so the gap is not explained by storage. The SGD head is not the best learner in this setting.
- On CLINC150, meta-preparation did not beat ordinary preparation or raw prototype memory under the predeclared criterion (paired differences of −0.05 and +0.01 pp against ordinary preparation, with changing sign across seeds). Prepared prototypes beat the meta-prepared neural learner on both suites in every seed mean. Gradient updates do add about 5.5 to 5.9 pp over first-example imprinting alone.
- Sequential forgetting is largely competition with newer labels, not demonstrated erasure. On BANKING77, original-label-only accuracy at the end (90.95%) was close to its initial value (90.50%), and 92.29% of final old-group errors predicted a later label. On CLINC150, restricting to the original labels recovered 98.22% on held-out domains. These restricted scores are oracle diagnostics and do not fix the deployed all-label task.

## Limitations
- The task is supervised classification into a supplied set of intents. It does not test language acquisition from scratch, label discovery, generative question answering, or general lifelong learning.
- The arrival orders are experimental. The data are a static dataset streamed in a chosen order, not a chronological customer feed. Concept drift, noisy feedback, and unknown-intent detection were not tested.
- The encoder was pretrained on unknown text, which may include these public datasets. Exact absence from pretraining cannot be established. Near-duplicate paraphrases remain.
- Five stream seeds are replications of one pretrained encoder, not five independent models. The SDs are descriptive, and the CLINC150 result rests on one fixed domain split.
- The CLINC150 criterion is a descriptive research gate, not a significance test. Offline preparation uses repeated sampling of the preparation categories, while runtime uses each support once.
- The BANKING77 development split was reduced from 16/20 to 8/12 per category after a count check, before any model selection. The follow-up diagnostic was declared after the main results and is labeled exploratory.
- Memory and state counts cover tensor payloads only, not Python or framework overhead. The deployment timings are descriptive desktop measurements.

## Key files
- `BANKING_EXPERIMENT.md`: predeclared BANKING77 protocol (data, learner, controls, selection, criteria).
- `BANKING_DIAGNOSTIC.md`: follow-up declared after the main results, testing whether forgetting is erasure or competition with new labels.
- `CLINC_EXPERIMENT.md`: predeclared CLINC150 protocol (domain split, meta and ordinary preparation, runtime evaluation, criterion).
- `results/banking77/REPORT.md`: BANKING77 results, diagnostic, selection table, resource accounting.
- `results/clinc150/REPORT.md`: CLINC150 results, paired contrasts, sequential retention, resource accounting.
- `clm/text_online.py`: online SGD classifier, prototype memory, nearest-neighbor memory.
- `clm/banking_data.py`: BANKING77 loading, duplicate filtering, development/support/query split, encoding cache.
- `clm/banking_evaluate.py`: learning-rate selection, online evaluation over both schedules, audit.
- `clm/banking_report.py`: generates BANKING77 tables and figure from raw results.
- `clm/banking_demo.py`: interactive teach/ask learner with saved sessions.
- `clm/clinc_data.py`: CLINC150 split (domains, categories, deduplication) and feature cache.
- `clm/rapid_model.py`: rank-32 adapter, expanding cosine classifier, imprinting, functional single-step updates.
- `clm/rapid_train.py`: ordinary and meta preparation of the adapter initialization.
- `clm/rapid_evaluate.py`: development selection, runtime evaluation, audit.
- `clm/rapid_verify.py`: independent reconstruction of final states and predictions.
- `clm/rapid_report.py`: generates CLINC150 tables and figure.
- `results/banking77/results.json`, `results/banking77/selection.json`, `results/banking77/diagnostic.json`: raw outputs.
- `results/clinc150/results.json`, `results/clinc150/selection.json`, `results/clinc150/conclusion.json`: raw outputs and paired comparisons.
- `STATUS.md`: running summary of findings and open questions.

## Sources
- `BANKING_EXPERIMENT.md`, `BANKING_DIAGNOSTIC.md`, `CLINC_EXPERIMENT.md` (protocols and their declaration dates).
- `results/banking77/REPORT.md`, `results/clinc150/REPORT.md` (all reported numbers in this log are copied from these files).
- `results/clinc150/conclusion.json` (per-seed paired differences), `STATUS.md`, `README.md`.
- BANKING77: PolyAI task-specific datasets (CC BY 4.0), revision `57ec275d8078af65b7731c2a98be812d844a6d6b`; Casanueva et al., 2020, https://aclanthology.org/2020.nlp4convai-1.5/.
- CLINC150: authors' repository https://github.com/clinc/oos-eval, revision `828f8093932c8fe6ca7936c3d2e52903b1c523de`; Larson et al., 2019, https://aclanthology.org/D19-1131/.
- Encoder: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- MAML, as cited in `CLINC_EXPERIMENT.md`: https://proceedings.mlr.press/v70/finn17a.html.
