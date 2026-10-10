# Executive control of retrieval

Implemented the first bounded retrieve-versus-act component from `e7b2f10`.
The [design](DESIGN.md) describes the broader hypothesis. The
[first protocol](PROTOCOL.md) and [prior-work review](PRIOR_WORK.md) preceded
execution. Read the [research decision](CONCLUSION.md) and
[complete run report](results/retrieve_act_20260910T224325010154Z/REPORT.md).

The first run stopped after development under its declared readiness gates.
The [instruction-delivery follow-up](results/instruction_delivery_20260910T225955352340Z/REPORT.md)
tested an interface explanation: all simple instruction controls passed, but
none of three delivery methods made evidence use reliable enough for confirmation.
Its [protocol](INSTRUCTION_DELIVERY_PROTOCOL.md) preceded all 92 local calls.
No held-out or confirmation cases were run. Preserve these tasks as component
regression fixtures rather than validated executive-control benchmarks.

[MODEL.md](MODEL.md) specifies the executive policy we intend to learn, its
observations/actions, and the proposed learning objective. The current model
policies are prompted baselines; no executive-control weights have been trained.

## Component boundary

`experiment.py` contains the inspectable components:

- `PassiveStore`: immutable snapshots and mechanical entity lookup in stable ID
  order. No outcome scores, authority filtering, relevance labels, or decisions.
- `Runner.trial`: applies a fixed policy or asks the decision model for
  `act`/`retrieve`, enforces the budget, invokes search, and logs actions.
- `Runner.generate`: loopback-only model calls with pinned digest, frozen prompts,
  constrained JSON, token accounting, and failure logging.
- `heuristic` / `resolve`: explicit baseline policies outside memory. The symbolic
  policy uses observable records, never target labels.
- `score`: evaluator-only scoring after the final action. Targets and case types
  are not included in model requests or sent to the passive store.

In this first isolation, the query and candidate limit are fixed by protocol.
The model owns whether to retrieve and how to answer from supplied evidence.
It does not yet write queries, select individual reads, or write memories.

## Local reproduction

From `D:\research\executive-control` with the existing Ollama service running:

```powershell
python -m unittest -v test_experiment.py
python experiment.py --run
python analyze.py results/retrieve_act_20260910T224325010154Z
```

The last command audits the original run without inference. Substitute a newly
created run directory to analyze a reproduction. Python standard library only;
no model downloads or environment installation. Reproduction creates another
development run and applies its stopping rules, rather than bypassing them.

The instruction-delivery diagnostic has its own frozen protocol and commands:

```powershell
python -m unittest -v test_instruction_delivery.py
python instruction_delivery.py --run
python analyze_delivery.py results/instruction_delivery_20260910T225955352340Z
```

Next prerequisite: isolate entity, authority, time-validity, and conflict handling
on fresh minimal pairs before composing them. Do not train a gate to compensate
for failures of an unreliable answer policy in a symbolically solved task.

All original MemRL artifacts remain in their original directory, unchanged and
closed as inconclusive. No MemRL records were used in these tasks.
