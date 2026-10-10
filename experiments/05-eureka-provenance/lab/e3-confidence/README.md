# E3 — the confidence head: does the model know it's about to fabricate?

First architectural component in the [fork](../gpt/README.md). E2's baseline
model fabricates a fact for 100/100 entities it never saw. This experiment asks
whether its own hidden state knows: we add a **detached probe head** —
`nn.Linear(n_embd, 1)` reading the trunk's last hidden state, trained with a
proper scoring rule (BCE) against on-the-fly labels of "was the LM's argmax
prediction at this position actually right?". Detached means no gradient flows
into the trunk: the language model is bit-identical in behavior to baseline;
the head is pure readout. (Toy cousin of semantic-entropy probes,
arxiv.org/abs/2406.15927.)

Trained on E2's **baseline** data (facts only, no abstention examples). At eval,
read the probe at the answer position:

- separation: mean confidence on known vs heldout answers, plus AUC
- **confidence-gated abstention**: refuse when the probe is below threshold —
  abstention from architecture, with zero calibration data, on a model whose
  *behavior* fabricates 100%.

Benchmark to beat: E2's data-level calibration (70/100 abstained, 100/100
known kept).

## Run

```powershell
cd ..\gpt
python train.py ..\e2-abstention\train_facts.py --compile=False --dataset=../nanogpt/data/facts_char --out_dir=out-conf
python ..\e3-confidence\eval.py out-conf
```

## Results

**Null — confirmed by diagnostic.**

| signal at answer position | known | heldout | AUC |
| --- | --- | --- | --- |
| probe confidence | 0.999 | 0.999 | 0.458 |
| LM's own max softmax prob | 1.000 | 1.000 | 0.479 |

Confidence gates at 0.5 and 0.9 block 0/100 fabrications. The second row is
the decisive control: the model's *own output distribution* is equally
saturated on memorized and never-seen entities — the fabrication is confident
all the way down. Nothing readable at the answer position marks the boundary
of this model's knowledge.

Read together with E2, this is the finding: the same trunk *can* represent
and use familiarity when training data gives it a reason (E2's calibrated run
acts on it), but a baseline trained only to predict text encodes no readable
self-knowledge for free. **Self-evaluation is not emergent at this scale — it
exists where training demanded it.**

Limits: linear probe, final layer, single position, char-level pure-memorization
regime. A nonlinear probe, earlier layers, name-token positions, or models with
genuine output entropy could still hide weak signal.

## E3b — joint gradients (`--conf_joint=True`)

Null again: AUC 0.521, saturated at 1.000. Routing confidence gradients into
the trunk can't help, because the objective is satisfiable without any
familiarity signal: **in training, this model is never wrong at an answer
position** — every queried entity is memorized. You can't learn to recognize a
failure state that never occurs.

## E3c — inject boundary states (one-shot distractor QA)

400 distractor QA lines placed once each in the corpus, so the model would have
genuinely wrong answer positions to learn "I'm wrong" from. Null in both
variants (detached AUC 0.500, joint 0.487; joint also dropped one known fact).
Diagnostic found the mechanism: **37/50 "one-shot" lines were memorized** —
2000 iters × 64 batch × 256 block ≈ 64 effective epochs over a 513K corpus, so
corpus-rare became training-frequent and the failure states evaporated; the
remainder drowned in the unweighted BCE.

## Series conclusion (closed after 3 attempts)

In the small-corpus memorization regime, correctness-supervised confidence is
untrainable: **a converged memorizer's training set contains no failures.**
Persistent failure states need data the model cannot memorize — explicit
held-out-entity examples (which is exactly E2, the mechanism that works) or an
open-ended data stream. Lesson for real systems: calibration training must use
fresh/held-out queries, never the corpus the model was trained on.
