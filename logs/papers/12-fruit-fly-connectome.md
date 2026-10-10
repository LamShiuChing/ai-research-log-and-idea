# Does Structured Sparse Recurrence Help Small Networks Learn? Controlled Comparisons Against Random and Degree-Preserving Wiring

Eric Lam (independent researcher, Toronto)

## Abstract
Fly connectome studies make it tempting to assume that measured wiring carries useful computation. We ask a narrower question: does a structured sparse recurrent mask help a small network learn, compared with random wiring and with degree-preserving rewiring? We built a PyTorch prototype (SRCN) with a synthetic module-and-hub topology and compared it with five controls on three memory and ordering tasks, and on an online reward-reversal task. Supervised results over three paired seeds show no consistent advantage for the structured topology. Dense recurrence was strongest on memory tasks but used three to four times the parameters of the sparse models. Ordering stayed at chance for all models. On the reward task, every sparse learner recovered after an unannounced reversal, and a learning-disabled control did not, so the benefit comes from online readout learning rather than from topology. Related experiments on a male-CNS connectome subgraph (flynet) were mixed and partly unverified; we report them with caveats.

## 1. Introduction
Recent connectome work links measured wiring to circuit predictions and to behavior in living flies (Shiu et al., 2024), and models visual responses with few parameters that rely on cell-type and spatial sharing (Lappalainen et al., 2024). These results show that wiring constrains function. They do not show that a sparse graph with biological-looking structure learns better than a comparable random graph. A connection probability and a rich-club organization are measured properties of one dataset (Lin et al., 2024), not inherited capabilities.

Recent preprints make the control question sharper. One reports that apparent topology benefits can vanish once initialization and node degrees are controlled (Dhiman, 2026), which motivates a degree-preserving rewired control. Another reports navigation robustness in a robotic setting (Wang and Chen, 2026), which we do not test here.

We ask: under matched parameter and edge budgets, paired seeds, and degree-preserving controls, does a structured sparse recurrent mask improve learning? The hypothesis is that structured wiring (`srcn`) beats random wiring (`random_sparse`) and rewired wiring (`rewired_sparse`) on learning and out-of-distribution generalization.

## 2. Methods

### 2.1 Models
All recurrent models use the update `h <- h + alpha * (tanh(input_projection(x) + W_sparse @ h) - h)`, with fixed heterogeneous time constants and a hidden width of 64. Only permitted recurrent edges are trainable. The SRCN graph has four synthetic modules, preferential local connections and a small hub population; edge density is 0.125 off the diagonal. `random_sparse` has the same width and edge count with uniform sampling. `rewired_sparse` applies directed edge swaps that preserve every node's in- and out-degree. `dense_rnn` has all off-diagonal recurrent connections with the same state width. `matched_dense` and `gru` are sized to match total trainable parameters (within 10%). The three sparse models share input and readout initialization, time constants and edge count per seed; the structured and rewired models also share initial edge values per destination.

### 2.2 Supervised tasks
- **Delayed recall:** remember one of 8 symbols across 4 distractor steps. Out-of-distribution (OOD) evaluation uses 12 distractors. Chance is 12.5%.
- **Rule switching:** remember the latest cue and use it to select one of two query bits across 4 distractors. OOD uses 12. Chance is 50%, and a memory-free strategy reaches 75%.
- **Transitive ordering:** shuffled adjacent facts from 6-node chains, with the order of two interior nodes queried. OOD uses 8-node chains. Chance is 50%.

A sensor-loss condition removes 25% of input channels for an entire episode and keeps the event markers visible. Training uses 200 optimizer steps, batch 32, learning rate 0.003, and 6,400 episodes per run, with identical episode streams within each task and seed. Each of the six models was trained on each task with seeds 0, 1 and 2 (54 runs).

### 2.3 Online reward task
Four cues and two actions with scalar immediate reward. The rewarded action mapping reverses halfway through without a signal. The recurrent core is frozen. Readout weights for the chosen action are updated by `delta = reward - Q(h, a)` times the presynaptic feature, with epsilon-greedy exploration at 0.1. Each run has 1,500 steps, with updates pooled across 32 streams. The control has the same observation format and no readout learning. Chance is 50%, and the optimal expected reward under epsilon 0.1 is 95%.

### 2.4 Analysis
Results are reported as mean ± sample SD across three seeds. Paired differences are SRCN minus control at matching seeds. The SD describes seed variability, not a confidence interval.

## 3. Results

### 3.1 Supervised tasks
Table 1 gives in-distribution (ID) and OOD accuracy.

**Table 1.** Supervised accuracy (%), mean ± SD over 3 seeds.

| Task | Model | Parameters | ID | OOD |
|---|---|---:|---:|---:|
| Delayed recall | srcn | 1,792 | 92.8 ± 3.1 | 46.2 ± 11.3 |
| Delayed recall | random_sparse | 1,792 | 92.4 ± 2.9 | 44.4 ± 13.8 |
| Delayed recall | rewired_sparse | 1,792 | 91.3 ± 4.7 | 40.2 ± 4.1 |
| Delayed recall | dense_rnn | 5,320 | 100.0 ± 0.0 | 95.4 ± 5.6 |
| Delayed recall | matched_dense | 1,810 | 96.2 ± 3.8 | 81.6 ± 11.5 |
| Delayed recall | gru | 1,826 | 97.4 ± 4.5 | 76.8 ± 32.7 |
| Rule switching | srcn | 1,082 | 81.9 ± 3.5 | 75.5 ± 3.2 |
| Rule switching | random_sparse | 1,082 | 80.1 ± 1.4 | 77.3 ± 1.0 |
| Rule switching | rewired_sparse | 1,082 | 81.8 ± 3.3 | 76.8 ± 1.3 |
| Rule switching | dense_rnn | 4,610 | 97.7 ± 1.4 | 63.7 ± 5.3 |
| Rule switching | matched_dense | 1,075 | 79.3 ± 1.8 | 77.9 ± 1.6 |
| Rule switching | gru | 1,067 | 83.3 ± 7.8 | 78.1 ± 3.4 |
| Transitive ordering | srcn | 1,850 | 47.9 ± 1.2 | 48.6 ± 4.7 |
| Transitive ordering | random_sparse | 1,850 | 48.8 ± 4.3 | 48.2 ± 2.5 |
| Transitive ordering | rewired_sparse | 1,850 | 48.0 ± 2.7 | 49.1 ± 3.2 |
| Transitive ordering | dense_rnn | 5,378 | 47.3 ± 3.2 | 50.7 ± 4.4 |
| Transitive ordering | matched_dense | 1,838 | 47.8 ± 0.9 | 47.5 ± 3.9 |
| Transitive ordering | gru | 1,923 | 49.5 ± 0.6 | 46.6 ± 3.8 |

Paired SRCN-minus-control differences are small and noisy. For delayed-recall OOD, SRCN minus rewired is +5.99 points (SD 15.05) and SRCN minus random is +1.82 (SD 6.72). For rule-switching OOD, SRCN minus random is -1.82 (SD 3.63). The 25%-sensor-loss condition shows no advantage either (delayed recall: SRCN 70.6 ± 4.1, random 72.1 ± 2.8, rewired 70.4 ± 2.6).

Dense recurrence was the strongest model on delayed recall, but it had three to four times the parameters of the sparse models. Rule-switching ID accuracy for the sparse models (about 80 to 82%) sits near the 75% memory-free reference, so these scores do not establish rule memory. Transitive ordering is at chance for every model.

### 3.2 Online reward task
**Table 2.** Percent rewarded actions during online learning, mean ± SD over 3 seeds.

| Model | Before reversal | Immediately after | Recovered |
|---|---:|---:|---:|
| srcn | 95.2 ± 0.5 | 7.3 ± 3.2 | 94.8 ± 0.2 |
| random_sparse | 95.2 ± 0.5 | 5.3 ± 0.3 | 94.9 ± 0.2 |
| rewired_sparse | 95.2 ± 0.5 | 5.3 ± 0.3 | 94.9 ± 0.2 |
| matched_dense | 94.8 ± 0.3 | 10.7 ± 9.2 | 94.1 ± 1.0 |
| srcn, learning disabled | 49.4 ± 0.3 | 50.2 ± 0.4 | 50.3 ± 0.9 |

All learning models recovered to about 94 to 95% after the reversal. The learning-disabled control stayed at chance throughout, so readout learning is responsible for recovery. No sparse topology separated from the others. An earlier actor-critic pilot did not recover (recovered accuracy 49.5 ± 0.4 for srcn and 49.5 ± 0.4 for random_sparse); it used a different algorithm and is not the current learner.

### 3.3 Related fly-derived experiments (flynet)
We also ran experiments on a 2,000-neuron subgraph of the male central nervous system connectome (178,570 edges), comparing the measured wiring with shuffled, random, dense and GRU agents on odor navigation, and with reward-gated plasticity on an odor-conditioning task. In odor conditioning, the real-wiring plastic agent reached late-trial accuracy 0.74 in its default run (chance 0.50), the shuffled plastic agent 0.51, and a dense RNN 0.51. Seed variation was large: the real-wiring seeds 1 to 3 gave 0.65, 0.71 and 0.61, and single seeds of a rule-off control and a dense model reached 0.74. A structured-mask experiment on an adding problem gave a final test MSE of 0.00043663440737873316 for the unsigned fly mask and 0.0012879511341452599 for a uniform random mask, in one run per mask. Working notes report that the learning-efficiency comparison found real wiring tied with dense wiring, shuffled wiring learned in 0 of 8 seeds, and that the structured-mask experiment found no supervised advantage. We could not reproduce the notes' criterion from the repository, and the single-seed mask result points the other way. We treat the flynet results as unresolved.

## 4. Discussion
In this controlled setting, structured sparse recurrence did not outperform random or degree-preserving wiring. The strongest supervised model was the dense recurrent network. Its advantage may reflect its larger parameter count, which we did not match. Matched-budget sparse and GRU models were close to one another, and every model was at chance on transitive ordering. The result matches the preprint caution that topology effects can disappear once degrees and initialization are controlled (Dhiman, 2026).

The reward experiment shows that a simple local rule, applied to readout weights, recovers from an unannounced reversal across topologies. This makes the reward result useful as a negative control for topology claims: a good result on an easy task does not require special wiring. The actor-critic pilot's failure shows that the algorithm, not the topology, is what needs to be right.

The fly-derived results do not change this conclusion. They are too few, too noisy and partly unverified to support an advantage for measured wiring. The connectome-level statistics reported in the literature are measured properties of a dataset with specific filtering (Lin et al., 2024). Whether they transfer to learning is a separate question that needs a separate test.

These findings are also consistent with the view that behavior-level predictions from connectomes rely on neuromodulation and cell-type specifics that a rate-based sparse network does not capture (Shiu et al., 2024; Bennett et al., 2021). Our models have no neuron types, Dale constraints, spikes, delays or neuromodulators, so they cannot test those mechanisms.

## 5. Limitations
- The SRCN topology is synthetic, not a measured fly connectome.
- Three seeds, one graph and one initialization draw per seed, a quick preset of 200 steps, and no per-model learning-rate tuning. Seeds are paired, but graph and initialization seeds are not crossed.
- Parameter budgets differ across the dense, matched and sparse models. Compute was not matched. Sparse parameters do not imply sparse execution speed.
- Validation and test episodes use independent random streams, so overlap can occur by chance.
- The reward task is an immediate-reward bandit with no delayed credit assignment.
- The flynet results come from single runs, mixed update budgets, and working notes whose criteria are not in the repository.
- No significance test is reported. Paired SDs are larger than many of the paired means.

## 6. Conclusion
Under paired seeds, matched edge counts and degree-preserving controls, structured sparse recurrence gave no consistent advantage over random or rewired wiring on three small supervised tasks or on an online reward-reversal task. Readout learning is what makes the reward task work. Testing whether measured connectome wiring helps requires more seeds, crossed graph and initialization draws, tuned baselines, delayed-reward tasks, and a provenance record for the measured data.

## References
- Bennett, J. E. M., et al. (2021). Nature Communications. https://www.nature.com/articles/s41467-021-22592-4
- Dhiman (2026). Topological Sensitivity. arXiv:2604.04033. https://arxiv.org/abs/2604.04033
- Lappalainen, J. K., et al. (2024). Nature. https://www.nature.com/articles/s41586-024-07939-3
- Lin, A., et al. (2024). Nature. https://www.nature.com/articles/s41586-024-07968-y
- MaleCNS v1.0 flat-connectome files. https://male-cns.janelia.org/download
- Shiu, P. K., et al. (2024). Nature. https://www.nature.com/articles/s41586-024-07763-9
- Wang and Chen (2026). FLYNN. arXiv:2607.00025. https://arxiv.org/abs/2607.00025
