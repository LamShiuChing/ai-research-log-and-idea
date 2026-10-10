# Online Adaptation of Frozen Sentence-Encoder Classifiers: Simple Prototype Memories Versus SGD Heads and Meta-Prepared Adapters on BANKING77 and CLINC150

Eric Lam (independent researcher, Toronto)

## Abstract

We ask whether a small classifier head on a frozen pretrained sentence encoder can learn new intent categories from one pass through labeled messages, and whether meta-preparation of a low-rank adapter improves this beyond ordinary supervised preparation and simple memory. On BANKING77, a zero-initialized 77-way linear head receiving one SGD step per example reaches 75.40% held-out accuracy after eight examples per category, compared with 1.29% for the frozen head and 1.47% when feedback labels are shuffled. Prototype and nearest-neighbor memories reach 80.05% and 80.54% under the same protocol and are more accurate. On CLINC150, a meta-prepared rank-32 adapter with an expanding cosine classifier does not beat ordinary preparation or raw prototype memory under our predeclared criterion. Gradient updates do add about 5.5 to 5.9 percentage points over first-example imprinting alone. Our results support supervised online adaptation of a classifier boundary on top of pretrained representations, not a learned learning algorithm, and they show that simple memory remains a strong control.

## 1. Introduction

Deployed text classifiers see new categories after training. A common response is to store examples and classify by similarity, or to update a small output layer. Both are cheap, but it is unclear how much a gradient step on one example contributes beyond what a stored average already provides. We studied this with a frozen MiniLM encoder (all-MiniLM-L6-v2) and compared several learners on two public intent datasets.

The study has two parts. BANKING77 tests whether a fresh linear head learns real customer messages online from one labeled example per category per update, and whether it retains earlier categories when new ones arrive. CLINC150 tests whether preparing an adapter by meta-learning improves single-pass acquisition of categories and domains withheld from preparation. Both protocols were written and dated before final evaluation. The CLINC150 result is negative under its own criterion, and we report it as such.

## 2. Background and scope

The encoder supplies language representations learned during its own pretraining, which we did not tune. All learners share identical encoder outputs. The experiments therefore measure what can be learned online from labeled feedback on top of fixed representations. They do not measure language acquisition from scratch, label discovery, or generative question answering.

The meta-preparation method is inspired by MAML (`https://proceedings.mlr.press/v70/finn17a.html`), but the implementation is an expanding imprinted classifier with a meta-trained adapter initialization, not a reproduction of the MAML benchmark. The CLINC150 dataset is from Larson et al. (2019, `https://aclanthology.org/D19-1131/`). The BANKING77 dataset is from Casanueva et al. (2020, `https://aclanthology.org/2020.nlp4convai-1.5/`).

## 3. Methods

**Encoder.** The encoder is frozen and run in evaluation mode with FP32 precision, attention-mask mean pooling, L2 normalization, a maximum of 256 tokens, and 384 output dimensions. No corpus statistics are fitted.

**BANKING77 learners.** The head is a zero-initialized, bias-free linear layer with 77 outputs (29,568 parameters). Each observation receives one cross-entropy backward pass and one SGD step, with no momentum, weight decay, replay, or extra epochs. Labels are per-stream permuted integers. We compare this with a frozen head (no updates), a shuffled-feedback head (same SGD, random labels), prototype memory (class sums and counts, cosine to class means), and nearest-neighbor memory (every observed embedding is stored). The learning rate was selected on development seeds 11, 22, and 33 from {0.1, 0.3, 1.0, 3.0, 10.0}; 0.1 was the smallest candidate and had the highest development score (73.47%).

**CLINC150 learners.** A residual rank-32 linear adapter (24,576 parameters) feeds normalized features into an expanding cosine classifier with logit scale 10. The first example of a category initializes its row from its own feature (imprinting, with no gradient). Each subsequent example causes exactly one SGD step on the adapter and all current class rows; the adapter step size is 0.02 times the head step size. Meta-preparation runs 600 Adam outer updates at learning rate 0.001, with four tasks per update, five categories per task, and four supports and four queries per category, differentiating through the online steps. The selected checkpoint is step 200. The ordinary-preparation control receives the same texts and outer steps, trains the adapter with a disposable 72-category supervised head, and discards that head before deployment.

**Data.** For BANKING77 we used the PolyAI task-specific release after normalized duplicate filtering, leaving 9,992 training-pool and 3,079 test messages. Development used 8 support and 12 query messages per category from the training split, giving 616 support and 924 query messages; 8,452 other training rows were eligible for final support. For CLINC150 we used the authors' release, excluded out-of-scope rows, and filtered duplicates, leaving 14,995 training, 2,998 validation, and 4,498 test rows. The category split is 72 preparation, 24 development, 24 novel categories in familiar domains, and 30 categories in two wholly held-out domains (meta and work), fixed by a torch generator seed of 719 and saved before training.

**Protocol.** BANKING77 used five stream seeds (101, 202, 303, 404, 505), with eight supports per category in the interleaved schedule and seven blocks of eleven categories in the new-category schedule. CLINC150 used three preparation seeds (17, 29, 43) crossed with five runtime streams and two suites, with four unique supports per category. Every support is used once in a lifetime. Predictions are recorded before feedback (prequential accuracy) and evaluated separately on held-out queries.

## 4. Results

**BANKING77.** Online SGD accuracy rose from 1.29% (no updates) to 52.21%, 60.68%, 69.77%, and 75.40% after one, two, four, and eight examples per category. At one example per category, the SGD head, prototype memory, and nearest-neighbor memory are tied (52.21%, 52.24%, and 52.24%). At eight examples, the paired SGD minus no-update difference is +74.11 ± 0.69 percentage points, and the paired SGD minus shuffled-feedback difference is +73.93 ± 0.74. The paired SGD minus prototype difference is −4.65 ± 0.86, and minus nearest-neighbor is −5.14 ± 0.92. Final accuracy in the new-category streams was 74.19 ± 0.62 for SGD, against 80.05 ± 0.92 for prototype memory and 80.54 ± 0.61 for nearest neighbor. Head reads took a median of 20.4 µs on CPU, and updates took a median of 127.9 µs. Persistent tensor storage was 118,272 bytes for the SGD head, 118,888 bytes for prototype memory, and 951,104 bytes for nearest neighbor.

**Retention.** In the new-category streams, the first group's full-candidate accuracy fell from 90.50 ± 3.87 to 65.76 ± 9.93 for SGD, and from 94.04 ± 2.05 to 80.27 ± 5.51 for prototype memory. A follow-up diagnostic, declared after the main results, restricted outputs to the first group's labels. SGD's restricted accuracy at the end was 90.95 ± 3.88, close to its initial value. Of its final old-group errors, 92.29 ± 4.18% predicted a later category. The restriction is an oracle diagnostic and is not a deployable score, but it indicates that the main decline is dominated by competition with newer labels.

**CLINC150.** On familiar-domain novel categories, meta-prepared neural accuracy at one, two, and four shots was 85.52, 92.04, and 95.56. The ordinary-prepared neural learner scored 86.23, 91.79, and 95.25; the raw prototype scored 85.86, 92.00, and 95.14. On held-out domains, meta-prepared neural scored 71.77, 77.00, and 84.15, against 72.09, 76.84, and 83.95 for ordinary preparation and 72.04, 78.80, and 85.76 for the raw prototype. The primary acquisition contrast (the mean of the one-, two-, and four-shot accuracies, averaged within each preparation seed) was −0.05 ± 0.90 percentage points versus ordinary preparation on familiar domains and +0.01 ± 0.58 on held-out domains. Its sign varied across preparation seeds, and the predeclared criterion was not met.

Prepared prototypes beat the meta-prepared neural learner on both suites in every seed mean. The meta neural learner minus its own prepared prototype was −0.20 ± 0.16 on familiar domains and −0.81 ± 0.03 on held-out domains. Removing gradient updates (imprint only) left accuracy flat at 85.56% and 71.75%, so the learned gradient steps contribute +5.48 ± 0.63 and +5.89 ± 0.17 points. In the sequential schedule on held-out domains, the meta neural learner ended at 76.19% over all introduced categories, against 85.76% for the raw prototype. Its first group fell from 99.37% to 52.18%, but restricting predictions to those labels gave 98.22%.

## 5. Discussion

The BANKING77 result supports the narrow claim that a fresh 77-way head can learn from one pass through real messages on top of a frozen encoder, and that its gain depends on correct labels. The shuffled-feedback control removes it. The comparison with memory is less favorable: prototype and nearest-neighbor memories are more accurate at every shot count from two examples onward, and the learned head is not the best use of the same representations here.

The CLINC150 result separates three effects. Meta-preparation did not reliably change acquisition over ordinary preparation. Averaging features in prototype memory, with or without meta-preparation, matched or exceeded the neural update. The neural update itself does improve over first-example imprinting. The sequential results suggest that much of the apparent forgetting is competition with later categories: restricted-label accuracy recovers to 98.22% on held-out domains, even though full-candidate accuracy is 52.18% for the first group. Because the restricted scores use information a deployed system would not have, they indicate where the loss comes from rather than offering a fix.

On storage, the SGD head (118,272 bytes) and prototype memory (118,888 bytes) are nearly identical on BANKING77, while nearest neighbor needs 951,104 bytes. On CLINC150, the raw prototype is smaller than the neural learners (37,056 versus 135,360 bytes at 24 classes). The raw CLINC150 prototype, which shares the smallest persistent state with raw linear SGD, also has the highest final held-out sequential accuracy among all learners tested (85.76%).

## 6. Limitations

The task is supervised classification into a supplied set of intents. Arrival orders are experimental, and the datasets are static, so the streams are not chronological customer traffic. Concept drift, noisy feedback, unknown-intent detection, and natural-language generation were not tested.

All learners share one pretrained encoder, whose pretraining data are unknown and may include these public datasets. Near-duplicate paraphrases remain in both datasets. The five BANKING77 streams and three CLINC150 preparation seeds are replications over one encoder and one fixed CLINC150 domain split. The reported ± values are sample standard deviations, not confidence intervals, and the CLINC150 criterion is a descriptive research gate rather than a significance test.

The BANKING77 development split was reduced from 16/20 to 8/12 queries per category after a row-count check that preceded model selection. The BANKING77 follow-up was declared after the main results and is exploratory. Memory figures count tensor payloads, not Python or framework overhead, and timings are descriptive desktop measurements.

## 7. Conclusion

With a frozen MiniLM encoder, a one-step-per-example SGD head learns intent categories from real messages, reaching 75.40% after eight examples per category from a 1.29% starting point. Simple prototype and nearest-neighbor memories are more accurate in this setting. On CLINC150, meta-preparation of a rank-32 adapter did not beat ordinary preparation or raw prototype memory under our predeclared criterion. Runtime gradient updates do add to first-example imprinting. The next question is whether a learner can preserve old-versus-new class comparisons under single-pass updates and beat prototype memory on this task.

## References

- Casanueva, I., et al. (2020). BANKING77 intent dataset (PolyAI task-specific datasets, CC BY 4.0). https://aclanthology.org/2020.nlp4convai-1.5/ ; https://github.com/PolyAI-LDN/task-specific-datasets
- Larson, S., et al. (2019). CLINC150 dataset. https://aclanthology.org/D19-1131/ ; https://github.com/clinc/oos-eval
- MAML, as cited in the project protocol (`CLINC_EXPERIMENT.md`). https://proceedings.mlr.press/v70/finn17a.html
- all-MiniLM-L6-v2 sentence encoder. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
