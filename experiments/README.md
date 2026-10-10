# Experiment code

Core code and reports from five of the experiments in the [research log](../README.md). Each folder is
copied from my working repo with data, checkpoints and environments removed. Most experiments follow the
same rule: write the prediction and the pass/fail bar first, then run.

| Folder | What it tests | Start here |
|---|---|---|
| [01-sleep-continual-gpt](01-sleep-continual-gpt) | A GPT that sleeps on its own dreams; running-sum memory over 20-day streams | `README.md`, `FINAL_REPORT.md` |
| [02-fact-memory-slots](02-fact-memory-slots) | Writing facts into memory slots on a frozen transformer; 2-hop reasoning | `HANDOFF.md`, `glt/` |
| [03-plasticity-claims](03-plasticity-claims) | Re-testing Dohare et al. (Nature 2024): 13 claims with falsifiers | `README.md`, `CLAIMS.md` |
| [04-executive-control](04-executive-control) | Should a local 14B LLM decide when to retrieve memory? | `README.md`, `CONCLUSION.md` |
| [05-eureka-provenance](05-eureka-provenance) | Knowledge, source trust and skills outside a frozen core (19 experiments + paper) | `README.md`, `paper/eureka.pdf` |

Also public, as separate repos:
- [LM-optimization](https://github.com/LamShiuChing/LM-optimization): 31 findings on how small GPTs learn reasoning
- [Language-Model-Optimization](https://github.com/LamShiuChing/Language-Model-Optimization): ELM, a LoRA-tuned local tool agent

Datasets are not included; each folder's docs say where they come from.
