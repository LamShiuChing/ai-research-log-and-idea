# E11 — conversation as an adapter

Is assistant-ness a *skill* in the accretion sense — installable as a module
on a frozen core? We trained zero-init bottleneck adapters (2.37M params,
1.9% of the model) on the frozen GPT-2 124M base, using 8,657 context-free
instruction–response pairs from databricks-dolly-15k rendered as
`User: …\nMind: …` dialogue. 3,000 steps, loss 6.24 → 2.48, ~4 min on the
RTX 4080. The serving app auto-detects the checkpoint and switches its chat
path to the adapted model.

## Before / after (same questions, same frozen base)

| question | raw GPT-2 (document continuation) | + chat adapter |
| --- | --- | --- |
| What is the capital of France? | rambling continuation | **"Paris"** |
| how do I make coffee? | genre drift | on-topic brewing instructions (semi-coherent) |
| tell me about the moon | "the moon? It was just me and you" | assistant-*shaped* answer, confabulated content |

Verdict: the **genre** of conversation — answer the question, stay on topic,
stop — transferred into a 1.9% module on untouched weights. Content quality
remains 124M-bounded (the moon answer confabulates), which is precisely the
gap the other organs cover (tool audit, memory, honest decline). The
ChatGPT-moment ingredient is, architecturally, an adapter.

## Run

```powershell
python prepare.py                       # downloads dolly-15k, builds BPE bins
cd ..\gpt
python ..\e11-chat\train_chat_adapter.py out-e11-chat
python ..\..\unified\serve_gpt2.py      # auto-detects the adapter -> :8343
```
