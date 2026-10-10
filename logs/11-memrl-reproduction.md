# MemRL reproduction (learned memory selection) on a local toy task
**Dates:** 2026-09-10 (all runs and the closing decision) · **Status:** concluded (closed as inconclusive) · **Hardware/scale:** one NVIDIA RTX 4080 SUPER (16,376 MiB reported), local Ollama 0.33.3, Qwen2.5 14B Q4_K_M for generation, Nomic Embed Text (768-dim) for embeddings; no cloud model calls. Scale: 16 held-out tasks per condition, four synthetic lab rules.

## Question
Does learned memory selection (MemRL utility scores over retrieved memories) improve task success over no memory and over similarity-only retrieval? The study was a small, local, synthetic adaptation of the MemRL paper, not a reproduction of its benchmark.

## Approach
- **Upstream:** MemTensor/MemRL at commit `c1b322ca43de36ddf64c6712f89d0095bfc35ce0` (MIT). One local patch makes the storage vector dimension configurable (default 3,072; this work uses 768).
- **Models:** Qwen2.5 14B Q4_K_M (generation) and Nomic Embed Text (embeddings), both local through Ollama, digests pinned.
- **Task (synthetic):** four fictional labs, each with a hidden arithmetic rule chosen from four alternatives by a seeded permutation. The model knows the alternatives but not the lab-to-rule mapping.
- **Method:** eight bootstrap attempts were stored as raw trajectories with the verified correct result. Utility (Q) learning used eight separate development tasks, twice, with binary success/failure feedback and no new memory entries. Memory and Q values were then frozen for 16 held-out tasks.
- **Conditions:** no memory; similarity retrieval (upstream retrieval, similarity weight 1, Q weight 0); MemRL hybrid retrieval (weights 0.5/0.5, learning rate 0.3, rewards +1/-1).
- **Follow-on studies (declared before each run):** a post-pilot development-only prompt diagnostic; a "direct-answer" memory-use baseline v2 (rule cards compiled from the bootstrap examples, with a fixed gate); and an "action" baseline in which the model picks a rule ID and a fixed calculator executes it.
- **Protocol:** temperature 0, fixed seeds, pinned digests, strict scoring, frozen memory checked before evaluation, controls for symbolic solutions.

## Results
| Test | Result | Baseline / control | Verdict |
| --- | --- | --- | --- |
| Initial pilot, 16 held-out tasks (direct answers) | No memory 4/16 (25.0%); similarity 1/16 (6.25%); MemRL 1/16 (6.25%) | No memory 4/16 | Null for MemRL. Both memory arms were worse than no memory: paired delta -0.1875 (-18.75 pp), 1 win / 4 losses per arm; cluster bootstrap 95% interval -0.5 to +0.125 |
| MemRL vs similarity retrieval (same pilot) | Identical selected memory lists on 16/16 tasks; paired delta 0.0 | Similarity 1/16 | Uninformative; the two arms were identical |
| Utility learning (same pilot) | All stored memories ended at Q = -0.7599 after the development trials, which all failed | — | Failed to discriminate between memories |
| Retrieval-presentation diagnostic (post-pilot, development tasks only, 8 tasks x 2 replays) | Original task-first prompt 0/16; same records with current task last 6/16; verified records only, current task last 6/16 | Original 0/16 | Exploratory. Suggests prompt placement matters on these eight tasks; not a held-out result |
| Direct-answer baseline v2, development gate | Validated rule cards 16/16 correct | Readiness required 15/16 | Gate passed |
| Direct-answer baseline v2, 32 fresh inputs (direct numeric answer) | No memory 2/32; verified examples 21/32; rule cards 22/32 | Readiness required 31/32 | Failed readiness; preserved as a failed attempt |
| Direct-answer baseline v2, correct rule with valid JSON | No memory 3/32; verified examples 24/32; rule cards 25/32 | — | Seven rule-card responses violated the JSON contract |
| Direct-answer baseline v2, fixed calculator (diagnostic) | No memory 3/32; verified examples 24/32; rule cards 25/32; fully symbolic 32/32 | — | Diagnostic only; the direct pipeline was the pre-selected score |
| Action baseline, development gate | Validated rule cards 16/16 correct, identical choices across replays | Gate required 16/16 | Gate passed |
| Action baseline, 32 fresh inputs (model + calculator) | No memory 8/32 (25.0%); verified examples 11/32 (34.375%); rule cards 32/32 (100%) | No memory 8/32 | Working control on this restricted task; cards vs no memory 24 wins, 0 losses, 8 ties; cards vs verified examples 21 wins, 0 losses, 11 ties |
| Action baseline, fully symbolic lookup + calculator | 32/32 (100%) | Rule cards 32/32 | Equal; the task needs no LLM |

Negative and null results are included above: the initial pilot, the failed direct-answer readiness gate, and the identical-retrieval comparison.

## What it showed
- The local memory loop ran end to end with the upstream MemRL storage, retrieval, and Q-update code. Frozen memory matched before and after evaluation.
- In the initial pilot, neither memory method beat no memory. Both memory arms scored lower (1/16 vs 4/16). The project's own report says this run is too small to distinguish the methods and does not show that memory harms models.
- Learned utility scores did not separate memories: every stored memory ended at the same Q value, so MemRL and similarity retrieval selected identical memories on all 16 tasks. The comparison therefore says nothing about whether utility learning helps.
- Given the same eight verified examples, a model that selects a validated rule and delegates arithmetic to a fixed calculator reached 32/32 on fresh inputs. A fully symbolic solver also reached 32/32, so this toy task does not show a need for an LLM or for learned selection.
- The project closed as inconclusive for learned memory selection and autonomous memory management (see the closing decision). The next step is to test whether the decision model chooses when to retrieve, in a separate project.

## Limitations
- Synthetic task with four fixed labs and four candidate rules; the 16 held-out and 32 fresh inputs are correlated within the four labs. Lab-level results are not independent replications.
- Not a reproduction of the paper's benchmark. The paper's BigCodeBench experiment was not run because Docker Desktop could not start reliably; the text pilot replaced it.
- Model, embedding model, memory format, and learning protocol all differ from the paper. The bootstrap gave the correct result (stronger than the binary reward used for Q learning).
- One model, one seed, and very small samples. The initial pilot's development set had eight unique tasks. Five of 16 identical request pairs returned different answers despite fixed temperature and seed, so exact reproducibility was not established.
- The post-pilot diagnostic was declared after the pilot's scores were seen and is exploratory.
- The action-baseline and direct-answer studies changed several things at once (response contract, execution policy, memory presentation, and input range), so their differences cannot be attributed to one factor.
- A simple program that matches examples to rules solves the task, which limits the interpretive value of any LLM result.
- Energy and GPU-seconds were not measured. The timings are not cold-start latency measurements, because Ollama reported cached prompt tokens.

## Key files
- `README.md`: project status, the three results, reproduction commands, and local setup.
- `CONCLUSION.md`: closing decision (inconclusive) and the handoff to the next design.
- `UPSTREAM.md`: pinned upstream commit, license, the local dimension patch, and setup steps.
- `TEXT_PILOT_PROTOCOL.md`: declared protocol for the initial memory pilot (conditions, limits, feedback).
- `DEVELOPMENT_DIAGNOSTIC_PROTOCOL.md`: declared protocol for the post-pilot development-only prompt diagnostic.
- `BASELINE_V2_PROTOCOL.md`: declared protocol for the direct-answer baseline with rule cards and a readiness gate.
- `ACTION_BASELINE_PROTOCOL.md`: declared protocol for the action-selection baseline with a fixed calculator.
- `experiments/text_pilot.py`: runs the initial pilot (no memory, similarity, MemRL) using the upstream memory service.
- `experiments/development_diagnostic.py`: runs the development-only presentation diagnostic on saved memories.
- `experiments/analyze_text_pilot.py`: recomputes the pilot audit from saved results without model calls.
- `experiments/baseline_v2.py`: runs the direct-answer rule-card baseline with its gate.
- `experiments/action_baseline.py`: runs the schema-constrained action baseline with the fixed calculator and symbolic control.
- `experiments/audit_action_baseline.py`: recomputes action-baseline outcomes and provenance from saved results.
- `experiments/local_memory.py`: local adapters and initialization for the upstream memory service.
- `experiments/local_smoke.py`: checks connectivity to the local models and the upstream providers.
- `configs/local_models.json`: pinned local model identifiers and digests.
- `results/text_pilot_20260910T204529522975Z/`: initial pilot, diagnostic, audit, and memory snapshots.
- `results/baseline_v2_20260910T215636160032Z/`: direct-answer baseline v2 (failed readiness).
- `results/action_baseline_20260910T220008778310Z/`: action baseline (32/32 with rule cards).

## Sources
- `README.md`: status, first-result and action-baseline numbers, local setup, and reproduction commands.
- `CONCLUSION.md`: summary table of all three studies and the closing decision.
- `UPSTREAM.md`: upstream commit and local patch.
- `TEXT_PILOT_PROTOCOL.md`: pilot design and limits.
- `DEVELOPMENT_DIAGNOSTIC_PROTOCOL.md`: diagnostic design.
- `BASELINE_V2_PROTOCOL.md`: baseline v2 design and gate.
- `ACTION_BASELINE_PROTOCOL.md`: action baseline design and gate.
- `results/text_pilot_20260910T204529522975Z/REPORT.md`: pilot scores, Q values, diagnostic table, failure inspection, resource counts.
- `results/text_pilot_20260910T204529522975Z/summary.json`, `audit.json`: pilot scores and paired contrasts (cross-checked).
- `results/baseline_v2_20260910T215636160032Z/REPORT.md`: direct-answer baseline v2 table and readiness result.
- `results/action_baseline_20260910T220008778310Z/REPORT.md`: action baseline table, paired counts, and gate result.
- `results/action_baseline_20260910T220008778310Z/summary.json`, `audit.json`: action-baseline scores and paired counts (cross-checked).
