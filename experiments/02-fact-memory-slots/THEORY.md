# GLT Theory

**Date:** 2026-08-19
**Depends on:** PROJECT_GOAL.md

- **§1–§7 — Note 1: Why Facts Cannot Be Baked In (And What To Do Instead)**, rev. 2. Revises
  PROJECT_GOAL.md §5 and §11.2; revises G5. Rev. 2 narrows §1's verdict — promotion by *replay*
  survives, only surgical weight-editing is ruled out (§2, §4) — adds §2 on biological precedent,
  and strengthens §3: the retraction limit appears **substrate-independent**, which changes what
  G5 should measure (§3.2). §7.6–§7.8 record what building it then falsified.
- **§8 — Note 2: Three Specialists and One Workspace.** Written after Note 1's design was built
  and measured (PROJECT_GOAL.md §14–§17). Composition over slot-stored facts failed structurally
  rather than by degree, which splits the single model into language, memory and inference
  specialists sharing the context window as a workspace.

---

## Read this part if nothing else

We wanted the model to learn a fact by writing it into its own weights, and to unlearn it
later by editing those weights again.

**Three separate lines of published evidence say that plan does not work.** Not "is hard" —
does not work, with numbers, reproduced across labs.

But the reason it fails tells us exactly what to build instead. The problem is not that the
knowledge is *inside the model*. The problem is that it is *mixed into the same parameters as
everything else*. Those are different things, and we had been treating them as one.

The fix: knowledge lives **inside** the network — in the forward pass, differentiable, native —
but in **its own parameters**, disjoint from the backbone. Inside, but not blended in.

> **Retractability requires parameter disjointness, not externality.**

That single distinction turns §11.2's fork into a measurable experiment instead of a judgement
call. Two further findings shape it:

- **The brain reached the same design, for the same stated reason, in 1995** (§2). Sparse
  non-overlapping hippocampal codes are the biological form of disjoint slots. This is
  convergent evidence, not decoration.
- **No known system — biological or artificial — can exactly retract a consolidated belief**
  (§3.2). Humans show the same suppression-not-deletion signature as edited LLMs. That makes
  exact retraction the wrong target, and G5 has to be reformulated (§3.3).

---

## 1. Four failures that get confused with each other

Erede's intuition was "writing into the network recursively will make a mess." Correct — but
there are four distinct messes, and only some of them are ours.

| # | Failure | What actually happens | Bites GLT? |
|---|---|---|---|
| 1 | **Model collapse** | Train a model on its own output, repeatedly. Rare events vanish; the distribution's tails die and do not come back. | **Only via the derivation loop** — if we train on self-derived conclusions. Not the bake-in step. |
| 2 | **Sequential-edit collapse** | Write facts into weights one after another *by targeted weight surgery*. General ability falls apart. | **Yes — fatal to edit-based writing.** |
| 3 | **Erasure is fake** | Edit a fact "out" of the weights. It is still there and can be pulled back out. | **Yes — fatal to retraction-by-editing.** |
| 4 | **Localization doesn't localize** | Find where a fact lives, edit there. That location is not where you should edit. | **Yes — "point at the fact and delete it" has no reliable referent.** |

### The evidence

**(1) Model collapse.** Shumailov et al., *AI models collapse when trained on recursively
generated data*, **Nature**, July 2024 — "indiscriminate use of model-generated content in
training causes irreversible defects in the resulting models, in which tails of the original
content distribution disappear."
<https://www.nature.com/articles/s41586-024-07566-y>
*(Note: a 2025 Author Correction exists — <https://www.nature.com/articles/s41586-025-08905-3> —
read it before citing the paper in any writeup.)*

**(2) Sequential-edit collapse.** Thede, Roth, Bethge, Akata & Hartvigsen, *Understanding the
Limits of Lifelong Knowledge Editing in LLMs*, ICML 2025, arXiv:2503.05683 — accuracy of
"ROME, R-ROME, and MEMIT … rapidly degrades within the first few hundred updates, leading to
model collapse." Even WISE, built for scale, declines within 10K updates and converges back to
pre-update knowledge levels. At scale, dedicated editing methods "fall short even when compared
to straightforward continual finetuning."
<https://arxiv.org/html/2503.05683v1>

Corroborated independently: *Spectral Characterization and Mitigation of Sequential Knowledge
Editing Collapse* (arXiv:2601.11042) attributes the collapse to progressive disruption of the
dominant singular directions of the pretrained weight matrices.
<https://arxiv.org/abs/2601.11042>

> **A few hundred writes.** Continuous learning means millions. This is not a tuning problem.

**(3) Erasure is fake.** Basani & Chhabra, *Exposing the Illusion of Erasure in Knowledge
Editing for LLMs*, arXiv:2606.23276 (June 2026) — knowledge editing creates "superficial
suppression mechanisms that leave original knowledge structurally intact and recoverable."
Adversarial suffixes recover the pre-edit fact at **85%+** on the same model/method, ~75%
across models. The mechanism is stated plainly: a rank-*r* update "can only influence a rank-*r*
subspace, leaving the remaining (*d−r*) dimensions unaffected."
<https://arxiv.org/html/2606.23276>

**(4) Localization doesn't localize.** Hase, Bansal, Kim & Ghandeharioun, *Does Localization
Inform Editing?*, NeurIPS 2023, arXiv:2301.04213 — causal-tracing localization "does not provide
any insight into which model MLP layer would be best to edit"; a fact can be changed by editing
weights in a *different* place from where it is stored.
<https://arxiv.org/pdf/2301.04213>

Background for what we were assuming: Meng et al., *Locating and Editing Factual Associations in
GPT* (ROME), NeurIPS 2022, arXiv:2202.05262 — mid-layer MLPs behave as key–value memories.
That result stands; (4) says it does not license the edit-and-delete operation we wanted.
<https://arxiv.org/pdf/2202.05262>

### Verdict (narrowed in rev. 2)

> **PROJECT_GOAL.md §5's promotion is unsound *if implemented as targeted weight editing*.**
> Every result above tests a surgical method — one located, rank-limited update per fact.
> None of them tests interleaved, rehearsal-based consolidation.

Rev. 1 of this note over-generalised from those papers and declared promotion dead outright.
That was wrong, and §2 says why: the brain performs promotion continuously, by a mechanism these
papers do not evaluate. **What is ruled out is the surgery, not the migration.**

The distinction is sharp and worth stating precisely:

| | Surgical editing (ROME/MEMIT) | Replay consolidation (biological, and standard continual-learning rehearsal) |
|---|---|---|
| Writes per fact | one large targeted update | thousands of tiny ones |
| Interleaving | none — the fact alone | the fact **mixed with rehearsal of prior knowledge** |
| Localization needed | yes — and finding (4) says it fails | no |
| Evidence against it | (2), (3), (4) | **not tested by any of the above** |

Interleaving is the whole difference. Catastrophic interference is caused by gradient steps that
see only the new item; rehearsal exists precisely to prevent it.

---

## 2. Biological precedent — the same design, for the same reason, in 1995

This section is convergent evidence, not analogy. The argument below is computational; it was
made about neural networks first and applied to biology second.

### 2.1 Why there are two memory systems

McClelland, McNaughton & O'Reilly asked why the brain bothers with a hippocampus *and* a
neocortex rather than one system. Their answer came out of hitting catastrophic interference in
connectionist models: **fast one-shot learning and slow generalisation cannot share a
substrate.**

| | Hippocampus | Neocortex |
|---|---|---|
| Learning rate | one exposure | many, small |
| Representation | **sparse, minimally overlapping** | dense, heavily overlapping |
| Optimised for | this specific episode | what is generally true |
| GLT's term | **disjoint memory slots** | entangled backbone weights |

Sparse non-overlapping coding is the biological form of parameter disjointness. The dentate
gyrus goes to structural lengths to ensure two similar experiences recruit *different* neurons —
pattern separation — for exactly the reason §3 gives.

McClelland, McNaughton & O'Reilly, *Why there are complementary learning systems in the
hippocampus and neocortex*, **Psychological Review 102**(3) (1995) 419–457.
<https://pubmed.ncbi.nlm.nih.gov/7624455/>

### 2.2 Promotion is real, and it is replay

From the same paper's account: memories are first stored via synaptic change in the hippocampal
system; those changes support **reinstatement** of recent memories in neocortex; **neocortical
synapses change a little on each reinstatement**; remote memory rests on accumulated neocortical
change.

That is promotion — and it is nothing like a rank-one edit. Many tiny updates, driven by replay,
interleaved with existing knowledge. This is the source of the correction in §1's verdict, and
it is also the mechanism the continual-learning literature calls rehearsal.

**Consequence for GLT:** a promotion path may be reinstated in the architecture, but it must be
specified as *replay with interleaving*, never as located weight surgery. It is a later phase
(P5), and it must be gated on G1 not regressing.

### 2.3 Quarantine-then-promote already exists at the synapse

Frey & Morris's **synaptic tagging and capture**: a weak stimulus sets a short-lived "synaptic
tag" at the potentiated synapse. The tag alone decays. Plasticity-related proteins, synthesised
by a *separate and stronger* event, are then "captured" by whichever synapses still carry a tag —
and only those changes become durable.

Read as a protocol: **provisional write → time-limited hold → made permanent only if an
independent salience signal arrives in the window.**

That is PROJECT_GOAL.md §6's quarantine-and-promotion, in wetware, since 1997. It also makes
§6's "surprise is a routing signal, not a truth signal" concrete: biology gates durability on a
separate salience channel rather than on the content of the write.

Frey & Morris, *Synaptic tagging and long-term potentiation*, **Nature 385**(6616) (1997)
533–536, doi:10.1038/385533a0. <https://www.nature.com/articles/385533a0>

### 2.4 What biology does *not* hand us

Not a licence to copy mechanism. Artificial "neurons" are a 1943 cartoon retained because it is
differentiable and fast on GPUs, not because it is accurate — real dendrites compute
nonlinearly, learning is local plus neuromodulatory rather than backpropagated, and synapses are
created and pruned rather than only reweighted.

The transferable content is the **constraint** — fast specific learning and slow general
learning need separate substrates — not the implementation. Planes do not flap.

---

## 3. The actual tension, stated precisely

Why does baking in destroy retractability? Not by accident — the two goals pull on the same
knob.

**Setup.** Let θ be the model's parameters. For a fact φ stored in θ, define its **footprint**
S(φ) ⊆ θ: the minimal set of parameters whose modification removes φ.

**Exact retraction** of φ requires an operation on S(φ) that removes φ and leaves every other
belief ψ intact.

**Sharing** between φ and ψ means S(φ) ∩ S(ψ) ≠ ∅.

Now the two requirements:

- **Retraction wants zero sharing.** If S(φ) ∩ S(ψ) ≠ ∅, then touching S(φ) may perturb ψ.
  Perfect retractability ⟹ every fact has a private, disjoint footprint.
- **Consolidation wants sharing.** The *reason* to put a fact in weights rather than a lookup
  table is generalisation: shared representation is what lets the model answer questions about
  φ that were never written down. Zero sharing ⟹ the "weights" are a hash table ⟹ we have
  reinvented external memory with extra steps.

> **Conjecture (Retraction–Consolidation trade-off).** Retraction completeness *R* and
> consolidation benefit *C* are anti-correlated along a single axis: the degree of parameter
> sharing between stored facts. Neither extreme is useful. The design question is where the
> knee of that curve sits — and whether a usable knee exists at all.

Deliberately **not** an impossibility theorem. It is a curve, which means it is measurable,
which means it can come back negative. If *R* and *C* turn out independent — you can have both —
the conjecture is falsified and GLT is easier than feared. Either outcome is a result.

### 3.1 Methodological consequence: measure retraction adversarially

Finding (3) means asking a model "what is Anna's team?" and getting "I don't know" proves
nothing; the fact may sit one adversarial suffix away. G5 as written in PROJECT_GOAL.md §4 is
too weak — it must include a *recovery attempt*, not merely a compliance check. Concrete
requirement on the P0 harness.

### 3.2 The limit looks substrate-independent

Humans fail at retraction the same way, and — importantly — with the **same signature**.

Correct a person's misinformation after they have reasoned from it, and the original keeps
influencing their judgements. This is the **continued influence effect**, and it replicates.
One leading account is *retrieval competition*: misinformation and correction "are stored
concurrently and compete for activation at memory retrieval," with continued influence occurring
when the misinformation is retrieved by automatic familiarity and the correction fails to
inhibit it in time.

Set that beside finding (3) — knowledge editing yields "superficial suppression mechanisms that
leave original knowledge structurally intact and recoverable."

> **Same failure mode, two unrelated substrates: suppression that competes, not deletion.**

Selected entry points (effect and mechanism confirmed at abstract level; primaries not yet read
in full):
- *Mechanisms in continued influence: the impact of misinformation corrections on source
  perceptions*, **Memory & Cognition** (2023). <https://link.springer.com/article/10.3758/s13421-023-01402-w>
- Replication of Johnson & Seifert (1994). <https://www.sciencedirect.com/science/article/pii/S0001691823002706>
- Partial biological retraction does exist, via reconsolidation: *Impairing existing declarative
  memory in humans by disrupting reconsolidation*, **PNAS**. <https://www.pnas.org/doi/abs/10.1073/pnas.1218472110>
  Note the framing — *impairing*, not erasing. Retrieval makes a memory labile and rewritable;
  it does not make it deletable.

This does two things. It **raises confidence** in the conjecture — the trade-off is not an
artifact of transformers. And it **changes the target**: if no known system achieves exact
retraction, demanding it of GLT is demanding something with no existence proof anywhere.

### 3.3 G5, reformulated

PROJECT_GOAL.md §4 currently sets G5 as "no orphaned descendants of a retracted premise" —
which presumes exact deletion. Replace with a three-part measurement that is honest about §3.2:

| Part | Measures | Why |
|---|---|---|
| **G5a — behavioural** | fraction of derived consequences withdrawn under normal querying | the original test; necessary, not sufficient |
| **G5b — adversarial** | recoverability of the retracted premise and its consequences under elicitation probes | per §3.1; catches suppression masquerading as deletion |
| **G5c — competition margin** | how decisively the correction beats the retracted belief at retrieval | the substrate-honest metric: exact deletion is the limit case of an unbounded margin |

G5c is the reformulation that matters. **Report the margin rather than assuming deletion is
achievable.** A system with a large, stable, and *provenance-explained* margin is a real result
even if the old belief is still physically present — and it is a claim humans demonstrably
cannot make about themselves.

Whether a disjoint-slot design can achieve an unbounded margin — genuine deletion, not
competition — becomes the sharpest empirical question in the project. §2.1 predicts it should,
for slots, and should not, for the backbone.

---

## 4. What survives: disjoint, not external

The trap was treating "inside the model" and "mixed into the weights" as the same thing. Split
them and a fourth option appears that we had not been considering.

| Where knowledge lives | In forward pass? | Deletable exactly? | Generalises? |
|---|---|---|---|
| External DB + retrieval (RAG, Kumiho) | ✗ — text prepended | ✓ trivially | ✗ |
| Context window | ✓ | ✓ (it evaporates) | ✗ — dies with the window |
| **Disjoint memory parameters** | **✓ native** | **✓ by construction** | **partly — the *read* is learned** |
| Entangled backbone weights (ROME/MEMIT) | ✓ | ✗ (§1.3) | ✓ |

Row 3 is the target. Knowledge sits in parameters the network reads through natively — gradients
flow, no retrieval call, no external service, memory content shapes computation at every layer —
but writing slot *k* touches only slot *k*, so deleting slot *k* is exact.

**This is buildable, and the substrate already exists at scale.** Berges, Oğuz, Haziza, Yih,
Zettlemoyer & Ghosh, *Memory Layers at Scale*, arXiv:2412.09764 (Meta FAIR, Dec 2024) — trainable
key–value lookup layers adding parameters without adding FLOPs, scaled to 128B memory parameters;
"gains are especially pronounced for factual tasks."
<https://arxiv.org/abs/2412.09764> · lineage: *Large Memory Layers with Product Keys*,
arXiv:1907.05242 <https://ar5iv.labs.arxiv.org/html/1907.05242>

And the update discipline we need has been tried: *Sparse Memory Finetuning as a Low-Forgetting
Alternative to LoRA and Full Finetuning*, arXiv:2605.03229 — updates only a selected subset of
memory slots per task, explicitly to avoid catastrophic forgetting.
<https://arxiv.org/pdf/2605.03229>
*(Headline forgetting numbers not extracted — the PDF's tables did not decode. Verify before
citing quantitatively.)*

Neither line was built for retraction or provenance. That gap is ours.

### 4.1 The design principle

> **Retractability requires parameter disjointness, not externality.**
> A fact can be genuinely internal to the network and still be exactly deletable, provided its
> parameters are its own.

This reframes PROJECT_GOAL.md §11.2. The fork is not *parametric vs. symbolic* — that was the
wrong axis, and both options can sit on either side of the real one. The fork is
**entangled vs. disjoint**, and §1 has already decided it: entangled-by-surgery is ruled out.

### 4.2 Revised memory tiers

PROJECT_GOAL.md §5's three tiers, corrected:

```
  Working context   (attention window)      volatile, per-request
  Memory slots      (disjoint parameters)   <-- WRITE here, online, exactly deletable
  Backbone weights  (pretrained)            language competence; frozen during inference
        ^                                   |
        +---- replay consolidation ---------+   (P5 only; interleaved, never surgical)
```

Two changes from §5:

1. **Nothing is written to the backbone at inference time, ever.** Online writes go to slots.
2. **The promotion arrow survives, but is respecified.** It is offline replay with interleaved
   rehearsal (§2.2), not located weight editing. It is a P5 concern, gated on G1 holding, and
   it is the operation that converts a slot's retractability into a competition margin (§3.3)
   rather than a deletion. **That conversion is a one-way door and must be treated as one:**
   promote only what the grounding and corroboration policy is willing to make hard to unwind.

**Still unresolved:** bounded capacity. In rev. 1 this was a hole opened by deleting promotion;
promotion is now back, but on a slow offline path, so slot pressure between consolidation cycles
still needs an eviction policy. Biology's answer is that forgetting is *active* — a maintained
process, not decay by neglect. Open (§7.1).

---

## 5. Where this leaves the claim

Honest positioning:

- **Taken:** justification-tracked memory with transitive retraction, done externally. Kumiho
  Inc., *Graph-Native Cognitive Memory for AI Agents* (2026-02-23, company whitepaper, not peer
  reviewed) formalises it against AGM belief revision over a Neo4j property graph, and ships
  `AnalyzeImpact(k,d)` — the transitive dependency cascade.
  <https://kumiho.io/pdfs/kumiho_AI_cognitive_memory_paper.pdf>
- **Ruled out:** the same thing done by surgical editing of entangled weights. §1.
- **Open, and ours:** the same thing done in *disjoint in-model parameters* — where retraction
  is still exact, but the read is learned rather than retrieved.

The falsifiable question GLT now owns:

> **Does an in-model, disjoint memory tier buy anything over an external store with the same
> justification graph — and if so, does it survive the retraction requirement?**

If yes, the contribution is real and narrow. If no, the honest finding is that memory belongs
outside the network, which is a publishable negative result and saves everyone time.

A second, sharper claim is now available and worth stating separately, because §3.2 makes it
non-obvious: **a disjoint-slot system may be able to do something no human can** — retract a
belief and all its derived consequences with an unbounded competition margin. If that holds, it
is the cleanest demonstration in the project that the architecture is doing real work.

### Classical foundations (verified, adopt rather than reinvent)

- Doyle, *A Truth Maintenance System*, **Artificial Intelligence 12** (1979) 231–272 (also MIT
  AI Memo 521). Justifications, and belief revision when discoveries contradict assumptions.
  <https://groups.csail.mit.edu/medg/people/doyle/publications/>
  *(Fine mechanics — IN/OUT status, SL/CP justifications, dependency-directed backtracking —
  unverified against the primary: the DSpace scan has no text layer.)*
- de Kleer, *An assumption-based TMS*, **Artificial Intelligence 28**(2) (1986) 127–162.
  <https://dl.acm.org/doi/10.1016/0004-3702%2886%2990080-9>
- Alchourrón, Gärdenfors & Makinson, *On the Logic of Theory Change: Partial Meet Contraction
  and Revision Functions*, **J. Symbolic Logic 50**(2) (1985) 510–530.
  <https://philpapers.org/rec/ALCOTL-2>
- Constraint worth knowing before formalising: Flouris, Plexousakis & Antoniou (2005) — AGM
  contraction does **not** transfer uniformly to description logics; postulate K÷4 fails. Stay
  in propositional logic over ground triples, deliberately.
  <https://ceur-ws.org/Vol-147/26-Flouris.pdf>

---

## 6. The experiment this implies

Runs on one 4080. Uses the existing 4-minute shakespeare_char loop as the iteration cycle.

**Sweep the sharing knob.** Build a memory tier whose degree of parameter sharing between
stored facts is a single tunable — from fully private slots (one fact, one slot) to fully
shared (a dense adapter). At each setting measure both:

- **R — retraction completeness**, per the reformulated G5 (§3.3): behavioural withdrawal (G5a),
  non-recoverability under adversarial elicitation (G5b), and the retrieval **competition
  margin** (G5c). Report the margin; do not assume deletion.
- **C — consolidation benefit.** Accuracy on queries about the injected fact that were *not* the
  written form — paraphrases, reversals, compositions — measured against an exact-lookup control
  that has *R* = 1 by construction.

Plot *R* against *C*. Three outcomes, all informative:

| Result | Meaning |
|---|---|
| Clean knee | GLT has a design point. Build there. |
| Strict trade-off, no knee | Conjecture holds hard. Memory should be external — publishable negative. |
| No trade-off | Conjecture falsified. Better than expected; find out why. |

**Second sweep, once P5 exists:** run the same measurement *before and after* replay
consolidation. §4.2 predicts *C* rises and the *G5c* margin falls. That prediction is the
sharpest test of this whole note, and if consolidation does **not** cost margin, §3's conjecture
is in serious trouble.

**Prerequisite, unchanged:** P0. The synthetic fact universe must exist first or none of these
numbers mean anything (PROJECT_GOAL.md §4, "the evaluation trap").

---

## 7. Still open

1. **Bounded capacity between consolidation cycles.** Replay handles the long run (§2.2, §4.2);
   slot pressure in the short run still needs an eviction policy. Biology's hint: forgetting is
   an active maintained process, not decay by neglect.
2. **Reversal curse in slots.** A key→value slot is a *directed* edge, same as a gradient
   update — PROJECT_GOAL.md §8's forcing function is not resolved by disjointness alone.
   Candidates: dual-index every write, or hold relations symbolically and let slots hold only
   the embeddings.
3. **Derivation loop vs. model collapse.** Failure (1) is still live: writing self-derived
   conclusions back is recursive self-training. Interleaving with externally-sourced data is the
   standard defence and is the same mechanism as §2.2's replay — so §6's grounding ratio and the
   consolidation policy are one problem, not two. Still no principled value for the ratio.
4. **Formalising the footprint S(φ).** §3 assumes it is well defined. Finding (4) suggests it
   may not be, for entangled weights — which is arguably *why* they fail. For disjoint slots it
   is definitional. Worth stating carefully; it may be the load-bearing lemma.
5. **Is exact retraction the right target at all?** §3.2 says nothing achieves it. G5c makes the
   weaker target measurable, but the threshold for "a good enough margin" is unset, and setting
   it honestly — rather than to whatever we happen to hit — needs doing before P3.

6. **Addressing is a sharing channel, and S(φ) does not currently include it.** *(found while
   building P2, 2026-08-19)*

   §3 defines the footprint S(φ) as the parameters whose modification removes φ, and §4.1 claims
   disjoint slots make it definitional. Building the read path shows that is only true of the
   **values**. A slot is reached through a *learned key*, and the read weight is a softmax whose
   denominator runs over every slot — so deleting slot *j* perturbs the normalisation of every
   other lookup. Values are disjoint; addressing is not.

   This is not an implementation detail that can be tuned away. It surfaced as a hard
   constraint: normalising *within* the selected top-k instead makes the k = 1 case degenerate —
   a softmax over one element is identically 1.0, its derivative is exactly 0, and the keys
   receive **no gradient at all**, leaving addressing frozen at random init. Verified directly:
   `keys.grad` is 0.000000 at k = 1 and non-zero at k ≥ 2. So the maximally-disjoint end of the
   sharing axis either cannot learn to address, or must share a normaliser. There is no third
   option in this formulation.

   Consequences:
   - **S(φ) must be defined over keys *and* values**, and the key term does not vanish at k = 1.
     §7.4's worry — that S(φ) may not be well defined — bites at the *disjoint* end too, not
     only the entangled end as assumed.
   - **G5's R measurement cannot treat slot deletion as clean** even in the best case. There is
     a floor on cross-fact perturbation set by the addressing scheme, and it should be measured
     rather than assumed to be zero.
   - An addressing scheme with no shared normaliser (per-slot sigmoid gates, hard assignment
     with a straight-through estimator, hashing to fixed slots) would remove the channel. Each
     trades something else away; none has been tried here yet.

7. **A memory tier is made by removing the alternative, not by adding memory.** *(measured
   2026-08-19; PROJECT_GOAL.md §14)*

   §4.2's tier diagram says "WRITE here" over the slot row, as though where a fact lands were a
   property of the architecture. It is not. Trained end-to-end with 2048 slots available, the
   model put every fact in the backbone and left the slots **at their initialisation** — values
   norm 0.3179 against an init of ≈0.32, contributing 2.0% of the MLP's residual magnitude.
   Zeroing all 2048 slots changed atomic recall by **0.0000**. Composition was 0.918 with slots
   and 0.918 without, on a third more parameters.

   Gradient descent stores a fact wherever it is cheapest, and a backbone with spare capacity is
   always cheapest. So the tiering in §4.2 is not something the architecture provides; it is a
   **constraint that has to be imposed** — by freezing the backbone, by starving it of capacity,
   or by routing writes so they cannot reach it.

   This has a sharp consequence for §6's experiment. The sharing sweep measures R and C against
   the degree of parameter sharing *between stored facts* — but if no fact is stored in a slot,
   every setting of the knob describes an unused module. **§6 cannot be run on a
   read-path-only model**, and P2/P3 in PROJECT_GOAL.md §7 are not separable as assumed: there
   is nothing to measure sharing over until a write path exists.

   It also sharpens the conjecture itself. §3 treats sharing as the independent variable, but
   the prior question is *placement*: whether a fact is in the memory tier at all. Placement is
   not free — it is bought by denying the backbone the option, and whatever that denial costs in
   general competence is a term the R-vs-C trade-off does not currently include.

8. **The write path cannot be gradient descent.** *(measured 2026-08-19; PROJECT_GOAL.md §15)*

   §4.1 requires that writing slot *k* touch only slot *k*. Gradient descent cannot satisfy
   this, and the reason is not tuning but objective: gradient moves whatever reduces the loss,
   and a write phase is only ever shown the facts being written. It never sees a query whose
   correct response is *"stay silent"*, so nothing opposes the keys spreading to cover every
   query. Three variants were measured, all landing in the same place:

   | Write | online_recall | kept_atomic |
   |---|---|---|
   | softmax read, gradient over whole table | 1.000 | 0.243 |
   | softmax read, gradient masked to fresh rows | 0.665 | 0.234 |
   | sigmoid-gated read, gradient over whole table | 0.570 | 0.202 |

   The gate variant is the informative one. Its bias learned to **−7.17** — the model was
   trying to shut the memory up — and the keys simply grew until the sigmoid saturated anyway:
   mean gate 0.999 on queries the memory has nothing for, **zero** queries silent out of 128.
   Masking writes did not help either, which rules out address-space churn as the cause.
   Meanwhile the frozen backbone still answered those queries at 0.999 with the memory
   switched off, so nothing was ever forgotten — the memory was *overwriting* correct answers.

   What this leaves:
   - **Bounded addressing.** Unnormalised dot products let a key grow without limit until it
     matches everything. Cosine similarity over unit-norm keys and queries bounds scores to
     [−1, 1], making a threshold meaningful and un-inflatable.
   - **Direct writes.** Set a slot's key *to the query vector of the fact it stores*, rather
     than searching for it by gradient. Then a key sits exactly where its fact is and drifts
     nowhere, which is what §4.1 asks for and what gradient cannot give. It also makes a write
     O(1) rather than a training run — arguably what "learn a fact at inference time" should
     have meant all along.
   - **Negative examples.** Interleaving old facts into the write phase supplies the missing
     "stay silent" signal directly. Note this is §2.2's replay, arriving as a *correctness*
     requirement for writing rather than a consolidation mechanism — and it needs the old data
     on hand at write time, which a continually-learning system may not have.

---

## 8. Note 2: three specialists and one workspace (2026-08-19)

### Why the single-model design ran out

Note 1 argued that knowledge should live in disjoint parameters inside the network. Built and
measured (PROJECT_GOAL.md §14–§17), that design does two of its three jobs extremely well and
the third not at all:

| | storage | exact deletion | composition |
|---|---|---|---|
| facts in backbone weights | good | **impossible** | 0.918 |
| facts in disjoint slots | **1.000** | **1.000 of targets** | **0.009** |

The composition failure is structural, not a tuning gap. To answer *"Anna's mentor's manager"*
the model must look up `(mentor, Anna)`, obtain `Ben`, and only then look up `(manager, Ben)` —
**the second address is a function of the first lookup's answer.** A transformer computing the
whole thing in one forward pass has nowhere to put the intermediate, so the second address never
exists. Measured at 0.009 against a 0.0044 chance line, flat across 20,000 steps.

Two further facts frame the redesign. First, the composing arm is not exact either: 0.918 after
100k steps of grokking, collapsing to 0.124 out of distribution. Neural networks approximate;
they do not reach 1.0, and §6's gates ask for reliability the substrate cannot supply. Second,
the *exactness* we do have comes from the part that is not a neural network at all — a slot is
addressed by a hash and deleted by zeroing a row.

The conclusion is not that Note 1 was wrong but that it asked one organ to do three jobs.
Language, memory and inference have different requirements, and only one of them is a job
neural approximation is good at.

### The three specialists

| Specialist | Job | Why it suits the substrate |
|---|---|---|
| **Language model** | parse a question, name the relation and entity, render an answer | pattern completion over text — what transformers are reliably good at |
| **Memory slots** | hold facts; exact write, exact delete | already built; recall 1.000, deletion 1.000, no interference |
| **Inference engine** | chain facts, propagate confidence and provenance | chaining is two lookups; exact by construction rather than 0.918 |

### They connect through a workspace, not to each other

Pairwise wiring between specialists is what makes such systems brittle, and neither established
account of this problem does it. The [blackboard architecture](https://www.opentrain.ai/glossary/blackboard-system/)
has independent knowledge sources that only read and write one shared structure, with a separate
*control* component choosing who acts. [Global Workspace Theory](https://arxiv.org/pdf/2410.11407)
(Baars) makes the same claim about cognition: autonomous specialist modules interfacing through a
single workspace whose contents are **broadcast to all of them**.

For GLT the workspace does not have to be invented. **The context window already is one** —
anything written into it conditions every subsequent forward pass, and the slot addressing
already hashes from it. The shared currency is a fact expressed as tokens, which all three
specialists can read and write.

This is what dissolves the composition failure. The second address was impossible because the
intermediate did not exist; once a specialist *writes `Ben` into the workspace*, the address
`(manager, Ben)` is computable by the same hash that already works:

```
cycle 1   "Anna's mentor's manager?"        language: this needs (mentor, Anna)
cycle 2   + "Anna's mentor is Ben"          slots: address computable, return Ben
cycle 3   + "Ben's manager is Carol"        slots: Ben is now in context, so is its address
cycle 4   nothing new -> halt               language: answer "Carol"
```

Same mechanism as chain-of-thought, but the intermediate is written by whichever specialist
knows it rather than guessed by the language model.

### Six channels, not a pipeline

| from → to | what flows |
|---|---|
| slots → language | retrieved facts enter the **forward pass itself** — this is what keeps the design out of the RAG row of §4's table |
| language → slots | new facts, gated by the MDL admission test (bits of surprise) |
| language → inference | which facts are relevant; inference alone explodes combinatorially and the language model is the pruner |
| inference → language | derived facts enter the context and change the next prediction |
| slots → inference | the fact store to chain over — the *same* slots, not a copy |
| inference → slots | derived conclusions written back, with provenance ⚠ |

The last channel is where §7.3's model-collapse risk lives. It is also where **G5a stops being a
research problem**: if inference records which premises produced which conclusion, invalidating a
premise deletes its descendants by construction. Note 1 hoped retraction would fall out of
parameter disjointness; in this design it falls out of bookkeeping, which is a far stronger
guarantee than anything gradient-based.

### Three decisions this design does not yet make

1. **Control — who acts when.** Blackboard calls this the control component, GWT calls it
   attention; both name it as the hard part. Proposal: run every specialist that *can* fire, halt
   when a cycle adds no new fact. Cheap, terminating, and no learned controller to debug. Revisit
   only if it proves too slow.
2. **Does the language model learn to use the workspace, or ignore it?** It must be trained on
   examples whose answer is only reachable *after* intermediates appear, or it will confabulate
   in one step as it does now. This is a training-data design question and should be settled
   before any code.
3. **Does inference write derived facts to slots, or recompute them?** Writing is fast, risks
   collapse, and demands provenance. Recomputing is safe, slower, and keeps the slot table purely
   externally-sourced — which also caps §6's grounding ratio at 1.0 for free. Proposal: recompute
   by default; treat writeback as an optimisation to be justified by measurement.

### The line that keeps this from being prior art

Kumiho already does belief revision over an external graph with a language front-end. The
distinction that must survive is the §4.1 one: **facts live in network parameters and are read
in the forward pass.** Keep that, and the inference engine is a reasoning controller over
parametric memory. Lose it — let inference maintain its own fact table — and GLT is a
reimplementation of the prior art with extra steps.

### Consequence for the roadmap

P4's derivation loop stops being a capability scheduled after P3 and becomes the **integration
layer** the whole design rests on. G3 is unreachable without it at any amount of read-path or
write-path engineering, and G6 (exact arithmetic, no decay with depth) becomes plausible for the
first time, because exactness now comes from a component that is not approximating.

---

## 9. Note 3: what the working system changes in the theory (2026-08-20)

Note 2 proposed three specialists and a workspace. It is built and measured
(PROJECT_GOAL.md §19–§21). Three things in the earlier notes need correcting, and one thing
that looked like a coincidence turns out to be the same law twice.

### 9.1 §3's conjecture was aimed at the wrong component

§3 asked whether one substrate can generalise and forget exactly, and §17 answered no at both
endpoints — a grokked backbone composes at 0.918 and cannot retract; disjoint slots retract
perfectly and compose at 0.009. The conjecture is confirmed *as stated* and simultaneously
stops mattering, because exactness never had to come from the network.

With premises recorded on the board, retracting a fact withdrew **1.000** of its derivations
with **0.000** collateral, and the recorded premises matched the ground-truth entailment graph
**3,402 times out of 3,402**. The cascade is a set operation. No gradient is involved, so no
gradient-based limit applies.

What survives of §3 is the narrower and more useful claim: *a network cannot be made to forget
exactly*, so do not ask it to. Put the facts where deletion is an address, and put the
derivations where deletion is bookkeeping.

**§3.3's reformulated G5 should be read with this split.** Measuring a competition margin
rather than assuming deletion was the right response to a world where retraction means
suppression. It remains the right response *for the backbone*. For slots plus a board, deletion
is literal, and the margin question applies only to whatever the backbone still knows.

### 9.2 Addressing by structure, twice — the same result in a different costume

§16 found that hidden-state addressing collapses: two facts sit at cosine 0.88, a margin of
0.035, and no linear projection recovers the difference. The fix was to address by token
identity, which separates by construction.

§19 found the identical shape at the language boundary. A generative head scores **0.000** on
any content word held out of its training data — not low, zero — because a softmax over the
vocabulary can only emit a token whose output embedding has taken a gradient. The fix was to
emit *positions* rather than words, so the token is copied rather than named.

Both are the same statement: **a learned embedding is an approximation of identity, and identity
is not a thing you should approximate.** Where a system needs to say *which one*, the answer must
be an index, a hash, or a pointer — something discrete that optimisation does not get a vote on.
This now looks like the design rule the whole project has been rediscovering, and it predicts
where the next failure will be: any place a network is asked to *select* rather than to *judge*.

### 9.3 Generalisation comes from the training distribution, not from size

The parser was swept on three axes against the same held-out wordings. A hundredfold in
parameters (0.14M → 14.88M) and twenty-five-fold in steps (4k → 100k) each moved the number by
nothing. Wording diversity (4 → 32) moved it from 0.445 to 0.895. Every configuration fit its
training data at 1.000, so nothing was capacity-starved.

This is the same mechanism as φ in §12. Wang et al. found grokking speed tracks the
inferred-to-atomic ratio and is not driven by dataset *size*; here, generalisation to unseen
wordings tracks wording diversity and is not driven by parameters or compute. In both cases the
rule wins only when the training distribution makes memorising more expensive than generalising.

The consequence for the roadmap is a plain division of labour. **Language coverage is bought with
a training distribution nobody can reproduce on one GPU, so buy it** — a pretrained model as
specialist 1, which is contamination-safe here because it never sees or emits an answer and the
entities are nonsense syllables absent from any corpus. **Exactness is designed, not trained,**
so build that. The token interface makes the swap a drop-in: the language specialist's entire
contract is to emit the canonical string.

### 9.4 The workspace makes the ceiling visible, and it is not language

With extraction solved, the binding constraint moved somewhere the earlier notes did not look.
`manager of mentor of Anna is` expresses exactly one idea: follow single-valued relations from
one subject. Counting, reverse lookup, comparison, time, and negation are not expressible at
all — and a better language specialist would only produce queries the memory cannot answer.

So the next research is **the query language**, not the parser and not the backbone. Each new
operation has to preserve the property that makes this design worth having: the answer is exact
because the operation is not a neural approximation. That is also the honest route to G6 — a
transformer's arithmetic decays with compositional depth, and an exact operation does not decay
at all, because it is not approximating in the first place.

### 9.5 Exactness is bounded by how often an operation reads memory

§9.4 said the next research was the query language. Built (PROJECT_GOAL.md §22), it produced a
rule the earlier notes did not anticipate.

`follow` reads memory once and is exact. `reverse` has no address to compute -- the question
fixes the object, and the hash is keyed by subject -- so it asks the forward question of every
entity and keeps the hits. That keeps facts parametric, at one batched pass, and it inherits one
chance to confabulate **per entity read**: 0.401 invented names per true edge, cut to 0.032 by a
confidence floor and not further.

    an operation is exact to the extent that it reads memory once

This reframes G4. Calibrated abstention was filed under metacognition, a Track C nicety. It is
not: it is the precondition for every operation that scans, because a memory that answers a
question it has no fact for turns one confabulation into N. The exact specialists cannot repair
that -- they are exact *given their inputs*, and a scan's inputs are N approximate answers.

It also predicts where the ceiling moves next. Counting inherits the scan's error. Arithmetic
over counted quantities would inherit it again, and G6 asks for accuracy that does not decay
with depth -- which now depends on abstention rather than on the arithmetic being exact.

