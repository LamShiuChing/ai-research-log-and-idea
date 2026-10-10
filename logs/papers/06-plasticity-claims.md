# Re-testing Loss of Plasticity and the "Totally Curable" Forgetting Claim on Permuted MNIST: A Falsification Testbed

Eric Lam (independent researcher, Toronto)

## Abstract

Dohare et al. (2024) report that deep networks trained by standard backpropagation on a long sequence of tasks lose the ability to learn new ones, and that L2 regularisation, shrink-and-perturb and continual backprop mitigate this. A podcast discussion built on that work further claims that catastrophic forgetting is "totally curable" with the same ideas. We re-test these claims on a scaled-down online permuted-MNIST stream. Each claim was written with a refutation condition before its number was taken, and verdicts use three seeds with a two-of-three rule. Plain backpropagation shows a drop from its best window (0.043, 0.042, 0.039), and Adam is worse on every seed. L2 mitigates the drop without removing it. Shrink-and-perturb does not beat L2 at either noise scale tested. Continual backprop maintains plasticity against plain backprop, but its utility-based unit selection is no better than random reinitialisation. Continual backprop increases forgetting (0.640 to 0.680 against 0.342 to 0.390 for plain backprop), so the forgetting claim is falsified. Replay reduces forgetting to about 0.02. The claim that weight-norm growth alone explains the loss remains undecided at three seeds. We report all results, including the falsifications.

## 1 Introduction

Continual learning asks a network to keep learning from a stream of tasks without stopping and without losing what it has learned [S4]. Two failure modes are usually discussed: catastrophic forgetting, where new learning overwrites old knowledge, and loss of plasticity, where the network gradually stops learning new tasks at all. Dohare et al. (S1) report the second effect on a long online permuted-MNIST stream, show that L2 regularisation, shrink-and-perturb and continual backprop (CBP) restore learning, and propose CBP as a remedy. A later discussion (S2) describes catastrophic forgetting as "totally curable" with the ideas behind CBP. The discussion was available only as a blurb, so we treat that stronger reading as a claim to be tested rather than a stated result.

This paper tests the round-1 claims of a larger project. The project's method is to write down each claim together with the observation that would refute it, run a check, and record survivals and falsifications with the same care. The paper covers the first round only. Later rounds moved to language models and are not reported here.

## 2 Method

**Model and stream.** The learner is a ReLU MLP with three hidden layers of 256 units and a single output head, with no task label. The stream is online permuted MNIST. Each task applies a fixed pixel permutation to MNIST, and each training example is seen once in random order. Online accuracy is measured on each example before the update. The plasticity checks use 200 tasks of 10,000 examples. The forgetting checks use 10 tasks of 60,000 examples. The source paper uses 800 tasks of 60,000 examples and 2000 units per layer, with mini-batch 1. We use mini-batch 16 and scale the step size to match per example.

**Arms.** Plain backprop uses SGD with learning rate 0.05. Adam uses learning rate 1e-3. L2 adds weight decay 1e-4. Shrink-and-perturb adds noise 3e-5 (C4) or 1e-4 (C4b) after each step. CBP replaces low-utility mature units at rate 1e-4, with maturity 100 and utility decay 0.99; utility follows equation 1 of the paper, without the downstream bias compensation in the authors' code. The control arms are CBP with random unit selection (C11), a per-task norm reset to initial RMS (C10), CBP at rate 1e-5 (C8b), and a 2,000-example reservoir replayed 1:1 into each batch (C9).

**Claims and bars.** Each claim has a statement, a source, and a falsifier written before the verdict runs. Plasticity claims compare last-window accuracy (the mean over the last 10 % of tasks) with the best sliding window, and use a 1-point margin. Forgetting is the mean over earlier tasks of best-ever minus final test accuracy. Forgetting claims use a 2-point margin, and the naive-training control (C7) requires at least 10 points. Seed 0 served as a pilot that fixed the scale and never entered a verdict. Verdicts use seeds 1, 2 and 3, and a claim survives if it holds on at least two of three. Hyperparameters come from the paper's grid, scaled and recorded before running. None were tuned on the verdict seeds.

## 3 Experiments and Results

Table 1 gives the plasticity results. Values are seeds 1, 2 and 3 for the last-window accuracy. The parenthesised values are the best-minus-last drop.

**Table 1. Plasticity (200 tasks × 10,000 examples).**

| Arm / claim | Last window (best − last) | Verdict |
|---|---|---|
| BP (C1) | 0.828 (0.043) / 0.827 (0.042) / 0.831 (0.039) | SURVIVES 3/3 |
| Adam 1e-3 (C2) | 0.810 (0.050) / 0.815 (0.048) / 0.817 (0.046) | SURVIVES 3/3 (below BP on each seed) |
| L2 1e-4 (C3) | 0.844 (0.027) / 0.841 (0.030) / 0.847 (0.024) | SURVIVES 3/3 (mitigates, not eliminates) |
| Shrink and perturb, noise 3e-5 (C4) | 0.837 (0.034) / 0.845 (0.025) / 0.844 (0.027) | FALSIFIED 3/3 |
| Shrink and perturb, noise 1e-4 (C4b) | 0.840 (0.031) / 0.844 (0.027) / 0.847 (0.024) | FALSIFIED 3/3 |
| CBP 1e-4 (C5) | 0.865 (0.009) / 0.867 (0.005) / 0.869 (0.005) | SURVIVES 3/3 against BP; own-drop passes by 0.1 point on seed 1 |
| Norm reset (C10) | 0.845 (0.019) / 0.851 (0.014) / 0.856 (0.007) | UNDECIDED (n = 3) |
| Random reinit (C11, vs CBP) | 0.864 / 0.863 / 0.865 (CBP: 0.865 / 0.867 / 0.869) | SURVIVES 3/3 |

Table 2 gives the correlate and forgetting results.

**Table 2. Correlates (seed 1, plain BP) and forgetting (10 tasks × 60,000 examples).**

| Arm / claim | Result | Verdict |
|---|---|---|
| BP dead units, |w|, effective rank (C6) | Dead units 0.027 → 0.146; mean \|w\| 0.031 → 0.073; effective rank 41 → 51 | FALSIFIED 3/3 (rank rises) |
| Naive BP forgetting (C7) | 0.390 / 0.355 / 0.342 | SURVIVES |
| CBP 1e-4 forgetting (C8) | 0.640 / 0.680 / 0.660 | FALSIFIED 3/3 (CBP forgets more than BP) |
| CBP 1e-5 forgetting (C8b) | 0.435 / 0.466 / 0.433 | SURVIVES 3/3 (still worse than BP) |
| Replay 2000 forgetting (C9) | 0.024 / 0.023 / 0.021 | SURVIVES |

In Table 2 the forgetting values for BP are the baseline for C8, C8b and C9. The C7 baseline is the BP value itself.

## 4 Discussion

Three findings stand out. First, the drop in plasticity under plain backpropagation appears on this stream, and Adam is worse, which matches the source paper's direction. Second, the mitigations are not equal. L2 reduces the drop but leaves 2.4 to 3.0 points of decline from the best window. Shrink-and-perturb did not improve on L2 at either noise scale. Third, the utility-based selection in CBP adds nothing over random reinitialisation (C11). In the terms of the ledger, what maintains plasticity here is injecting fresh random weights and keeping norms small; the utility-based selection is the ingredient that is measurably harmful to retention (C8) and never measurably helpful (C11).

The forgetting results are the clearest falsification. CBP increases forgetting (C8). Its utility measure favours units that are active on the current stream, so units that serve only earlier tasks are reinitialised first. Lowering the replacement rate reduces the damage but does not reverse it (C8b). Replay reduces forgetting to about 0.02 (C9). The strong reading of the podcast claim is therefore not supported at this scale.

Two results do not match the paper's account. Effective rank rises under SGD (C6), and the rank collapse appears only under Adam. The paper describes dead units, growing weights and rank collapse together. Here the weight growth and dead units are present under SGD, but the rank collapse is not. The norm-reset claim (C10), which would make weight growth the mechanism, is undecided: two of three seeds exceed the drop bar by 0.4 to 0.9 points under the registered rule, but the ledger reports a seed spread of about 1.2 points for this statistic, which is larger than that margin, so the claim needs seeds 4 to 9 before a verdict.

Later rounds of the same project tested claims about retention economics and language models. Their verdicts are mixed and are not reported in detail here. The project's own summary is that a retention-based objective did not pay on the MNIST stream at any recurrence rate tested (C22), but did pay on a substrate where relearning is expensive (C25a).

## 5 Limitations

- The setup is scaled down: 256 units per layer, mini-batch 16, 200 tasks of 10,000 examples for plasticity, and MNIST only. The results may not transfer to the paper's scale or to other data.
- Each arm uses one hyperparameter setting, taken from the paper's grid and scaled. A falsification means "false at these settings". No sweep was run.
- CBP is implemented without the downstream bias compensation of the authors' code. Results apply to this variant.
- Three verdict seeds. C10 is undecided, and C5's own-drop component is a near-bar pass. Escalation to seeds 4–9 is required for both and was not run.
- The reported values are copied from the ledger's evidence table, not recomputed from raw results. Two in-text ranges in the ledger do not match its own table (C8's "27–33 points" versus the table's 25.0 to 32.5 point differences from BP; C5's "3.6–3.8 pts" versus 3.7 to 4.0). We report the table values.
- The source discussion (S2) was available only as a blurb, and the workshop paper (S3) was not retrievable, so the stronger reading of S2 is our reading, not the speakers' stated position.

## 6 Conclusion

On a scaled-down online permuted-MNIST stream, plain backpropagation loses plasticity, Adam is worse, and L2 mitigates the loss without removing it. Shrink-and-perturb did not improve on L2 at the two noise scales tested. Continual backprop maintains plasticity, but its utility-based selection performs no better than random reinitialisation. Continual backprop increases forgetting, which falsifies the claim that catastrophic forgetting is totally curable with its ideas, while replay is effective. The claim that weight-norm growth alone explains the loss is undecided. These results are at one hyperparameter setting on a reduced scale, and the escalation runs needed for the two undecided or borderline claims remain open.

## References

- [S1] Dohare, S., Hernandez-Garcia, J. F., Lan, Q., Rahman, P., Mahmood, A. R., & Sutton, R. S. (2024). Loss of plasticity in deep continual learning. *Nature*, 632, 768–774. https://www.nature.com/articles/s41586-024-07711-7 (read via https://pmc.ncbi.nlm.nih.gov/articles/PMC11338828/). Code: https://github.com/shibhansh/loss-of-plasticity.
- [S2] Sequoia, *Training Data*: "Rich Sutton and Khurram Javed: Why AI Models Stop Learning, and How to Start It Again." https://www.youtube.com/watch?v=xH7U7w9Qzlo. Only the episode blurb was retrievable.
- [S3] Javed, K., & Sutton, R. S. (2024). The Big World Hypothesis and its Ramifications for AI. *RLC 2024 workshop*. https://openreview.net/pdf?id=Sv7DazuCn8. Not retrievable; wording used only from secondary sources and unverified.
- [S4] IBM Think, "What is continual learning?" https://www.ibm.com/think/topics/continual-learning. Not read (HTTP 403). Used only as a generic taxonomy of continual learning methods.
