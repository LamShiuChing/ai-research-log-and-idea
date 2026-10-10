# One Shared Scorer for Answers and Memory Actions: Gains, Regressions and Failed Gates in a Small Decision Model

Eric Lam (independent researcher, Toronto)

## Abstract
We study whether one shared scorer, built on a pretrained text encoder, can choose typed answers and memory/tool actions from caller-supplied candidate descriptions. The model outputs choices, ordered scores and truth probabilities without generating text. We trained it on synthetic atomic tasks, public NLI, reading and causal data, and synthetic memory workflows. We ran matched comparisons for retention, and for supervised versus reinforcement-learning (PPO) training. Results were mixed. The model learned trained families well on new wording (371/384) and improved several reserved workflow and language scores. However, retention regressed (ticket routing fell by 59.4 points in one broad run), no candidate passed the predeclared retention gate, and in a single-seed comparison PPO trailed supervised training (133/192 versus 149/192). We report these results as limits on the approach, not as evidence of general reasoning.

## 1. Introduction
Many applications need a decision component that picks among stated options, reports probabilities and decides which tool or memory operation to perform next. A reference system with a public typed-decision interface motivated the output contract, but this project did not measure against that system. Our goal was internal: one shared model, trained once, that handles several decision types and whose failures can be measured. We focus on three questions. Does a shared candidate scorer learn new wording and names? Does broader training preserve earlier skills? Does reinforcement learning from verified outcomes improve on supervised action labels under the same budget?

## 2. Method

### 2.1 Model and output contract
The model reads state tokens with a pretrained encoder. A multihead attention layer lets each candidate read those state tokens. A shared MLP head maps each candidate to a scalar, and scores are normalized within a question. Question and option identifiers are opaque routing fields. The output is a choice, an ordered score or a truth probability. No token generation is involved.

Two backbones were used. The first, `all-MiniLM-L6-v2`, has 22.89M total parameters (22,894,593 measured). The second, a pretrained NLI DeBERTa encoder, has 141.53M parameters; its original classification head was discarded and replaced by the shared scalar scorer. A planning document had estimated the MiniLM backbone at about 22.7M parameters; we report the measured count.

### 2.2 Data
Training data came from four sources.
- **Synthetic atomic families:** logic conditions, ordering, lookup, events, routing, severity and helpdesk next action. Test families for conjunction and prerequisite selection were held out of training.
- **Public NLI:** WANLI pairs, with 30,000 balanced pairs in the generalization run and 12,000 three-way records in the broad run. Source groups were separated across splits.
- **Public reading and causal data:** BoolQ and QuaRTz. The grounded corpus contains 8,786 training records. Passage groups were separated across local splits.
- **Memory workflows:** arithmetic dependency graphs with supplied tools for integer arithmetic, freshness checks and invalidation. Workflows included timeouts and changed inputs. Reference planners produced demonstrations offline only; inference never called them.

### 2.3 Training and comparisons
Three comparisons define the study.
1. **Retention.** Two identical runs started from the same checkpoint with the same data, seed and update budget (2,400 updates). The treatment added a cached-reference KL term (weight 0.5, temperature 4). Reference logits came from the published model on training records, kept only where its argmax matched the label. The control had no reference term.
2. **Supervised versus PPO.** Both arms started from the same warm start (192 supervised updates). Each then received 512 episodes and 256 optimizer passes on four synthetic environments: document evidence, supplier table, inventory and task prerequisites. The supervised arm received accepted next-action labels for visited states. The PPO arm received +1 for a verified success, −1 for a wrong conclusion or exhausted budget, and −0.01 per action. PPO used a clip of 0.2, discount 0.99 and GAE λ 0.95.
3. **Calibration.** A single temperature was fit on a separate calibration split after weight selection.

Development data selected checkpoints. Reserved test sets were scored after selection. Several controls were reused after they were exposed, and we mark those results as non-fresh.

### 2.4 Gates
The retention gate was fixed in advance. A candidate could not be promoted unless no original-family or guided-helpdesk exact score fell more than five percentage points below the published model, and it had to improve broader inference and plain-observation recovery. Higher average scores alone did not qualify.

## 3. Results

**Learning trained families.** On reserved tests, the published MiniLM model scored 371/384 (96.61%) on six trained atomic families with new wording and names. A pretrained similarity baseline on the same families scored 160/384 (41.67%). Untrained conjunction and prerequisite families scored 37/128 (28.91%). Across all eight families the score was 408/512 (79.69%). Workflows with API defaults scored 12/24 (50%), against 0/24 for random actions. Removing name compilation raised this to 17/24 (70.83%).

**Calibration.** Temperature calibration reduced calibration-split NLL from 1.6476 to 0.4116. Final atomic NLL was 0.5140, Brier 0.2938 and ten-bin calibration error 0.1023. Calibration does not change the argmax choices, so it does not change correctness on new domains.

**Model size and accuracy.** On a dense logic diagnostic (128 training records, 50 presentations each, 400 updates), MiniLM reached 144/256 (56.3%) on development, with NLL 1.621. DeBERTa reached 209/256 (81.6%), with NLL 2.099. The larger model was more accurate but had worse probability quality. The two runs differ in pretraining, initialization and pooling, so this does not isolate the cause.

**Broader training and retention.** A MiniLM run of 1,600 updates raised reserved three-way NLI from 194/600 (32.33%) to 269/600 (44.83%) and plain workflows from 1/16 to 4/16. It lowered the six original families from 183/192 (95.31%) to 161/192 (83.85%), and severity from 32/32 to 15/32. Balanced accuracy on Boolean conditions was 48.53%, below the baseline's 50.49%. Raw accuracy of 82.50% was misleading, since always answering false scores 85%.

A matched retention experiment gave a clear negative result. The control lost 31.25 points on routing and 9.375 on lookup. The KL arm improved language and plain-workflow scores (for example, 274/600 on three-way NLI and 9/24 plain-observation workflows) but lost 21.875 points on severity. Neither arm passed the gate. KL was selected on development data and not promoted.

**Broad DeBERTa run.** After 6,000 updates, reserved memory workflows rose from 23/48 to 48/48 and ordering from 26.7% to 100%. Grounded yes/no reached 74.0% (from 51.7%) and qualitative causal questions 83.9% (from 47.2%). However, original ticket routing fell from 32/32 to 13/32, a 59.4-point regression. Routing also misclassified all ten duplicate-payment cases and all nine product-information cases. The run was not promoted.

**SFT versus PPO.** From the same warm start, the warm start scored 102/192 across the four adapted environments, supervised corrective training 149/192 (77.6%) and PPO 133/192 (69.3%). On a diagnostic environment omitted from adaptation, all four models scored 0/48, each concluding without gathering evidence. This is a single seed, a small synthetic study, and a fixed budget.

**Research preview.** A later supervised-repair preview scored 152/240 on a supplemental cohort, of which 77 were correct abstentions. Its document environment fell from 48/48 to 45/48, which exceeded the retention allowance, so the gate failed. Three authored free-form decision probes scored 2/3 for the baseline and 0/3 for the preview.

## 4. Discussion
The results support a narrow claim. A shared candidate scorer with explicit, typed outputs can learn new wording within trained families and can complete controlled multi-step workflows when the runtime supplies tools and memory. It does not yet support a general claim. Three findings are most consequential.

First, retention was the binding constraint. In every broad run, gains on new tasks came with regressions on earlier skills, most clearly ticket routing (−59.4 points) and severity (−21.875 points for KL). A reference-distribution term (KL) reduced some regressions and improved language scores, but failed the predeclared gate. A higher average score was not evidence of retention.

Second, accuracy and calibration disagreed. The larger encoder was more accurate on the dense logic diagnostic but had worse NLL. Temperature calibration improved the calibration split but cannot correct a confident wrong answer, such as a 90%-probability misrouting. Calibrated probabilities should not be read as correctness guarantees.

Third, the PPO result is a null under this budget, not a verdict on RL. The supervised arm used accepted-action labels on states the model visited, and PPO used sparse verified rewards. With one seed and 256 optimizer passes, PPO trailed supervised training (133/192 versus 149/192).

## 5. Limitations
- The corpora are small and synthetic, or public datasets with templates. The routing training set contained four distinct problem wordings, so row counts overstate the number of independent examples.
- Many reserved controls were exposed and then reused. Helpdesk results cover 44 scenario variants built from 14 ticket sentences. They are correlated and are not independent incidents.
- Most comparisons use one seed. The DeBERTa and MiniLM runs differ in several respects, so architecture effects are not isolated.
- Pretraining overlap cannot be excluded: MiniLM's training data includes AllNLI, and the NLI backbone's data includes SNLI, MultiNLI and WANLI's prompt provenance.
- WANLI and BoolQ class priors are imbalanced or balanced by construction, and related records are correlated. Statistical claims are therefore limited.
- The runtime supplies tool results, freshness checks and exact arithmetic. The model does not learn these.
- We reviewed, but did not run, a reference PPO implementation. The review found target-bearing observations and reused validation data. We did not reuse that setup.
- Serving is on CPU with two threads, so timings from CPU and GPU runs are not comparable.

## 6. Conclusion
A small shared decision scorer learns trained families well and can improve reserved workflows, but in this study, broader training repeatedly traded away earlier skills. No candidate passed the predeclared retention gate, and calibration did not fix wrong answers on new domains. PPO did not beat supervised corrective training under a fixed single-seed budget. The next experiment should preserve retention with fresh held-out wording and structures, and test whether evidence-gathering can be learned rather than relying on supplied descriptions. Until then, the published checkpoint remains an internal prototype.

## References
Only sources named in the project documents are listed. Full bibliographic details were not recorded in those documents.

- Proximal Policy Optimization (PPO). arXiv:1707.06347. https://arxiv.org/abs/1707.06347
- Generalized advantage estimation (GAE). arXiv:1506.02438. https://arxiv.org/abs/1506.02438
- WANLI dataset (generated and human-labeled NLI pairs). https://github.com/alisawuffles/wanli
- BoolQ dataset. https://github.com/google-research-datasets/boolean-questions
- QuaRTz dataset. https://huggingface.co/datasets/allenai/quartz
- all-MiniLM-L6-v2 model card. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- cross-encoder/nli-deberta-v3-small model card (diagnostic reference). https://huggingface.co/cross-encoder/nli-deberta-v3-small
