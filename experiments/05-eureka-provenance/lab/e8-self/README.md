# E8 — the self-learning loop

Can the system notice a skill it lacks, learn it, verify it, and register it —
autonomously? `controller.py` runs the whole loop; ground truth comes from an
executable arithmetic oracle (a general calculator tool, not task knowledge).

## The run (2026-08-31)

1. **Self-calibration**: novelty threshold set from the system's own geometry
   (max familiar-question distance 2.45 → τ 2.70). No hand-picked constants.
2. **Detection**: in a 200-question stream, flagged 53 as "not one of my
   skills" — **all 50 multiplications, 3 false alarms**. The base never needed
   to know what multiplication *is* to know it wasn't familiar.
3. **Synthesis**: derived the pattern and number ranges from the flagged
   queries; generated 6,000 oracle-verified training examples.
4. **First attempt — self-rejected**: 64-dim adapter reached 36/100, below
   the system's own 50-threshold → **adapter discarded by its own gate**.
5. **Escalation**: retried at 128-dim / 5,000 steps → 73/100, colors intact
   50/50 → accepted.
6. **Registration**: router retrained with the new class from the collected
   queries.

| final system (routed) | score |
| --- | --- |
| colors (base) | 100/100 |
| heldout addition (adapter B) | 63/100 |
| heldout subtraction (adapter C) | 100/100 |
| **multiplication (self-taught adapter D)** | **74/100** |

Nothing forgotten; the base untouched; a bad attempt caught and deleted by
the system itself; capacity escalated by policy, not by a person.

## Honest limits

- **Ground truth is the bottleneck**: the loop closes fully only where an
  oracle exists (executable/verifiable domains). Elsewhere it needs a teacher
  or human signal — the anchor problem, again and always.
- **The representational floor**: detection and learning both ride the frozen
  base's representations; a skill below the base's floor (unseen primitives,
  E7 v1) can be flagged as strange but not learned by any adapter.
- Escalation ladder is finite and hand-designed (two rungs); resource
  budgeting for autonomous training is an open policy question.

## Run

```powershell
cd ..\gpt
python ..\e8-self\controller.py out-e7-a out-e7-adapter out-e7-adapter-c
```
