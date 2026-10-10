# THESIS — Grown, not given (v0.6, 2026-09-01 — post-C23: the decay law demoted to a weak prior;
point 3 revised the same session, as E1's falsifier required)

The argument, stated once. `CLAIMS.md` holds the claims and numbers. Every commitment names what
would kill it. When evidence moves, this file moves in the same session.

## Position

Rejected: the frozen model — a system whose weights never change after deployment does not learn.
Problem statement, not a claim.

Accepted, in one reading: the big world hypothesis — the world is too large to model, so learning
never finishes and tracking beats convergence. An argument for lifelong updating, not against
memory: a buffer or audit the size of the machine is not a model of the world.

## The thesis, five points

**1. A small base model that continues learning by itself.** Small at birth and *growable* when it
saturates — capacity is added, never bought up front. Knowledge, including primitive coverage, is an
output of running, not a prerequisite. What is given is only the machinery of learning. The
architecture is open; the MLP is the current instrument, not the thesis.

**2. It learns from a stream — and the stream is the verifier.** The base's permanent background job
is prediction. No external judge at first contact: the model predicts, reality answers, error
judges. (The eureka gate was needed because stored facts had no stream to test them; here everything
kept is re-tested by living.)

**3. It does not decide what is worth keeping — it forgets. And forgetting happens where knowledge
still has an address.** Admission is free; eviction is the policy. Two kinds of forgetting:
- *Chosen* forgetting — decay of the unused, unconfirmed, contradicted. Choice needs an address, and
  inside a trained network knowledge has none (a fact is smeared over shared weights) — which is why
  every directed in-base deletion fails (CBP utility, C8; machine unlearning generally requiring
  retraining — from memory). An **unmerged adapter is an address**; so is a store entry. The decay
  law is therefore not a mysterious neural invention but a policy over an addressable pool with
  counters — uses, confirmations, contradictions, age — J's priced retraction (T2/T2b), ported.
- *Accidental* forgetting — interference at training time; catastrophic forgetting (C7), what CBP
  doubled (C8). Not a decision; handled mechanically (rehearsal + audit, point 4).
**The base never forgets by choice; choice happens before the merge, where knowledge is still
addressable. In-base forgetting is only undirected hygiene** (weight decay, norm renormalisation —
C10; biology's sleep-time synaptic downscaling is the same move, from memory) **plus slow drift of
what sleep never rehearses** (measured: C15i).
**Revision after C23 (E1's falsifier fired, 2026-09-01):** the address half of this point is
proven machinery — C15 and C17 run on it. The *decision* half is demoted: counter-priced eviction
beats random by only ~1.4 pts even in a Zipf world, captures under half of the clairvoyant's own
thin edge, and carries zero signal where recurrence has no structure (C23a/b/d; the signal is
lawful — C23d — just small). Operative truth at this scale: **bounded pool + eviction under
pressure; the choice of victim is low-stakes because relearning is cheap.** The counters stay as
bookkeeping and a weak prior, never a judge (the v0.4 external-objective position survives its
standing test). The scale caveat was then TESTED and removed (C25b, the facts world): priced
eviction loses to random even where relearning is expensive, because occurrence-weighted
admission already implements the frequency prior — the stream prices, the evictor has nothing
left to add. The demotion is unconditional on this instrument family.
Goal in one line, amended: forgetting has been turned from an accident into a decision (the
mechanism exists and is safe) — but at this scale the decision is nearly free to get wrong.

**4. Learning happens in a nursery: fast adapters on probation, then merge of the survivors.** The
stream trains a small fresh adapter (fresh weights are maximally plastic, C11; the mechanics now
validated on this instrument — identity-init all-layer low-rank + gates learns fresh domains at
84–90 % and repeated spawn→merge of one domain *converges*, C15g); the adapter lives *addressable*
for a probation period, its counters exposed to the decay law; most adapters die there (noise
self-clears: never reconfirmed → dropped — now proven: C29a 3/3, noise never graduates while
knowledge always does; and C29b 9/9, even unfiltered noise merges nearly harmlessly — the gate is
hygiene on top of an intrinsically robust sleep). Only survivors are merged in **sleep** — distillation
against a rehearsal mix, bounded O(machine), **anchored in stored confirmed answers where the store
has them** (C15j: ground truth retains ~1 pt better than self-behaviour targets — the audit
ledger's invariant, arrived at independently by experiment); weight addition is dead on this
substrate too (C15d: 63–73 pts lost, even when the fold is exact), and so is synthetic rehearsal
(C15e: 18–24 pts; F2 measured). **Merge is the event where knowledge gives up its address** — after
it, only undirected hygiene applies. What survives of the old "gate" is an **audit, not a judge**
— now tested (C17): its expectations come from the stream's confirmed answers, never the base's
past (the snapshot-anchored alternative demonstrably erases confirmed corrections, C17b);
rewrite-before-merge lets corrections through intact (C17a); rotation detection is exact at ≤ R
sleeps on unhealed damage (C17c); and detection turned out easier than feared — real drift is
bursty, so the per-item test sees it and the Page-Hinkley aggregate is a false-alarm-free second
witness rather than the last line (C17d falsified, informatively). Repair of rehearsal-covered
content is mostly free: the sleep's own rehearsal heals it, audited or not. Costs, owned and now *measured*: the **novelty tax** — ~0.1 pt of old-domain accuracy
per genuinely-new domain consolidated, which stops when the world stops being new (C15b falsified
as registered; the accelerated arm shows the tax is per-novelty, not per-sleep — drift does not
compound with time); unrehearsed knowledge erodes fast (C15i: 12–23 pts over 20 sleeps — rehearsal
scope is a real boundary, C17's canary watches it); merge damage scales with how much was learned
awake (probation dial). Dead dangers: base stiffening did not occur over 20–60 sleeps (C15c 9/9,
plasticity *improved*); "rebirth only in sleep" is falsified (C15f: ρ = 5 % random reinit costs
4–5 pts even with immediate rehearsal — C8's lesson holds asleep too) and, per C15c, unnecessary
at this horizon.

**5. No "when to learn" exists.** Learning is the resting state — always-on prediction — so nothing
triggers it. Needs enter later as a *scheduler* (where to point among streams: C20,
progress-with-habituation vs the twice-unbeaten uniform), not as a trigger. Randomness has exactly
one confirmed role: rebirth for plasticity (C11). The full embodied drive is sequenced after sleep
is proven; its failures would be uninterpretable while consolidation leaks.

## The objective is external (v0.4, decided after C22/C22b)

There is no internal score by which this system may judge its own progress. The candidate that
could have worked — one currency, total prequential bits, retention paying for itself through reuse
(`notes/design-currency.md`) — was given a fair, pre-registered test and failed it twice: on a
recurring world, replay visibly retained knowledge and saved ≈ 1 % of bits against a 10 % bar, at
two recurrence rates (C22 falsified 3/3, C22b survived 3/3 as predicted). At this scale relearning
is too cheap for memory to pay in bits; the scale caveat is recorded in CLAIMS.md, not leaned on.

Post-C25 nuance (2026-09-01): the currency's underlying *bet* — retention pays iff relearning is
expensive — is now confirmed on both sides (C22 falsified it where relearning is cheap; C25a
proved it at 25 % bits saved under scarcity). **Post-C31 (2026-09-02): and on language — the
same replay organ saves 24–31 % of lifetime bits on a recurring-corpus char stream at 85M
(C31 SURVIVES), closing the loop: three substrates, one law, and the MNIST rounds' scale
caveats are now readings, not hopes.** This does not revive the currency as an internal
objective (that failure was about self-judging, and E1's verdict is now unconditional — C25b);
it does mean the thesis's memory organs have proven economic ground in the band they were built
for.
Consequences, owned: **the objective is the external probe suite; every bar is a convention we
register and defend.** The mechanism is autonomous; the evaluation is not — the stream judges
content, the probes judge the system, and every internal signal (counters, decay, utility) is
bookkeeping that must forever prove, against external checks, that it beats random (the E1
control). Free constants are set by engineering against probes and registered like
hyperparameters; nothing derives them. This position is itself falsifiable: an internal signal
that predicts retention-value better than random would overturn it, and finding one would be a
major positive result.

## Commitments and their killers

**T1 — Plasticity is cheap.** Fresh random weights + small norms; no utility measure. (C5, C10, C11.)
Killer: a stream where reinit + norm control fail while a selective method succeeds.

**T2 — Interference cannot be decayed away; merging needs rehearsal and bounded memory, O(machine).**
Killer A: a no-rehearsal merge preserving old behaviour (weight addition already failed, E13; a
math/architecture-level solution — orthogonal or sparse updates — would be a *welcome* killer and
belongs in the ledger when attempted). Killer B: forgetting growing without bound despite a fixed
buffer (C13).
**Status after C13 (2026-09-01): the bounded buffer's cost is a measured law, not an unknown** —
gap to an unbounded store ≈ 0 until the world clears the buffer (knee ≈ 1.5×B on every seed),
then ~2.4 pts per world-doubling, with the bounded store still beating memoryless everywhere
tested (C13a). The sleep-side spot-checks fell under their 2× bars after escalation (C13b 0/9,
C13c 3/9): starvation there is real but mild. T2's bet holds — fixed memory degrades gracefully
and at a knowable rate.

**T3 — The stream judges content; decay judges retention at the adapter tier; the audit judges
surgery.** No internal truth-judge anywhere (every one tried failed: E3 ×3, E12a, CBP utility
C8/C11) and no external truth-gate at first contact. Killer: knowledge classes the stream never
re-tests rotting invisibly — if they matter, a store must hold them, and T3 narrows to predictive
knowledge.
**Status after C30 (2026-09-01): the killer was measured and did not fire at this scale** —
φ_store(H\*) = 0.98: the stream re-tests ~98 % of kept knowledge before it can rot (C30a), so the
store stays a tool, not an organ. Scale-scoped: re-measure φ on the language substrate. Same
round, the moving-world test: stale rehearsal did NOT go anti-knowledge (C30c, the program's
second missed predicted-fail — a store that still helps outweighs its stale tenth) and
contradiction-aware refresh beats no-memory by ~5 pts on drifted content (C30b) — the audit's
stream-refresh rule, profitably ported down to the store.

**T4 — Sleep inherits the whole problem and survives it.** Old behaviour within noise, plasticity
intact, over ≥ 20 consolidations. Load-bearing; if it fails, the architecture fails. (C15, C17.)
**Status after C32 (2026-09-02): T4 survives its first test at LM scale** — 20 LoRA-nursery
wake/sleep merges on the 85M char-GPT cost ≤ ~0.1 bpc of old-domain knowledge (C32a, 5/6 after
escalation; margins thin against the strict bar, recorded), the tax stays per-novelty (C32b),
and the weight-addition control reproduces C15d's catastrophe at 10× the bar (C32c FALSIFIED
3/3 as predicted — rehearsal-free LoRA-merge is an overwrite, not a consolidation; the
industry-relevant verdict).
**Status after C15 (2026-09-01): survived its core** — retention within 2 pts on 9/9 seeds,
plasticity intact 9/9, transfer faithful, spawn→merge convergent — with amendments: the novelty
tax is the honest per-new-domain cost (C15b falsified as registered), rehearsal must be grounded
and real (C15j, C15e), rebirth stays out of sleep (C15f), and what rehearsal never covers is not
protected (C15i). The audit half now holds too (C17 a/c survive; b/d falsified — b as designed,
d in the audit's favour): T4 stands at this scale, with its costs and boundaries measured.

**T5 — Coverage is grown; structure is given.** The tiny growable core on a survivable curriculum
acquires the primitives it needs. Killer: growth stalling where a pre-loaded core succeeds on the
same stream.

**T6 — Tools count.** The learner plus its external tools is the unit that must learn. Carried; not
under test this round.

## Open research named by v0.3

- The **decay law**: the policy over the adapter pool and its counters — thresholds, pricing,
  end-declarations (J T2/T2b is the symbolic reference; the port is ours to design).
- The **doorman** (C16): new task vs returning task vs continuation — spawn, wake, or keep training.
- **Probation length**: merge-early vs merge-late — the composition-vs-safety dial (a C15 arm).
- **Growth**: the saturation signal and mechanism (zero-init units; audit-failure rate as trigger?).
- **Habituation**: efficiency organ (don't pay to learn noise before forgetting it); later.

## Order of work

1. ~~**C15 — sleep.**~~ Done (2026-09-01): a/c/d/e/g/h survive, b/f/i/j falsified — see CLAIMS.md.
2. ~~**C17 — audit.**~~ Done (2026-09-01): a/c survive, b/d falsified — see CLAIMS.md.
3. ~~**C16 — doorman.**~~ Done (2026-09-01), humbling: detection and the A3 traps work (C16d 3/3,
   one spawn per noise segment, no hard-rule re-alarms; stream marginal-proof, C16c), but the
   routing VALUE mostly evaporates at this scale — always-continue lands within 1–1.7 pts of the
   doorman because relearning is too cheap (C16a/b falsified; C22's economics a third time), and
   short segments break it as predicted (C16e). Identity purity is bad (0.14–0.31): the decay law
   must not key counters on doorman identities here.
4. **Decay law** — design now unblocked: C15's counters and numbers exist. E1 (counters vs random
   eviction) is the standing control.
5. ~~**C21 composition**~~ Done (2026-09-01): the gap dissolved — read-only taps compose within
   1–2 pts of fine-tuning on the rule stream while the committed deep design trails 3–7 pts
   (C21a/b both falsified; primitives confirmed necessary, C21c). Adapter type follows novelty
   type: deep for input-shift (C15), tap for rules over known inputs. C16's doorman inherits the
   routing choice.
6. **P2 stream with shared primitives** → C18 coverage curve, C19 growth; **C20 scheduler** after C15.

Round 1 (C1–C11 + C4b, C8b) is the ground: plasticity thesis true as diagnosis, cure =
fresh-randomness + small norms, utility harmful to retention, replay the only surviving retention
mechanism — hence T2.

## Rules (unchanged)

Falsifier before number; seed 0 pilots, verdicts on 2 of 3 fresh seeds; hyperparameters registered
before running; nulls recorded like wins; one instrument, arms only — no restarts.
