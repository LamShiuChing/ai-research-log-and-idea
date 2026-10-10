# E13 — the reversible merge

LoRA-form skill deltas (rank-16 on the fused qkv projection, 147K params,
1.4% of the model) trained on the frozen E7 base, then merged into the base
weights by pure arithmetic and unmerged again. Four claims, four rows:

| model | skill A (facts) | skill B (heldout addition) |
| --- | --- | --- |
| base | 100 | 0 |
| base + LoRA (live) | 0 | 73 |
| **merged (W += B·A)** | 0 | **73** |
| **unmerged (W −= B·A)** | **100** | 0 |

- **LoRA learns real skills** through a frozen base (73/100 held-out
  addition, matching bottleneck adapters).
- **The merge is behaviorally exact**: one matrix addition per layer
  reproduced the live-LoRA model identically.
- **The unmerge is a perfect undo**: subtracting the ledgered delta restored
  the original base exactly — weight-level rollback is real.
- **The merge tax is total without routing**: a merged (hence always-on)
  delta destroyed skill A completely, exactly like E7's unrouted adapter.
  Mergeability is not a license to merge — it is the *mechanism* that gated,
  verified, sleep-time consolidation gets to use, with rollback as its
  safety net.

Together with the verified buffer (unified/buffer.jsonl) this completes the
online-learning pipeline's parts: conversation → store (instant) → verified
buffer → micro-LoRA (minutes) → exam → ledgered merge (milliseconds,
reversible). Every stage fast for its layer; every transition gated.

## Run

```powershell
cd ..\gpt
python ..\e13-lora\lora_train.py out-e7-a out-e13-lora
python ..\e13-lora\merge_eval.py out-e7-a out-e13-lora
```
