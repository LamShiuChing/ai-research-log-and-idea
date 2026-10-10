# Prior work reviewed — 2026-09-10

This is a focused review for the binary retrieval gate, not a systematic survey
or a benchmark reproduction. Sources were read before experiment execution.

| Source | Overlap with this component | Consequence for the experiment |
| --- | --- | --- |
| Sumers et al., [CoALA, v3](https://arxiv.org/html/2309.02427v3), especially Sections 4–5 | Organizes agents around memory, internal/external actions, and decision procedures. | Separating passive storage from action selection is an architectural choice with prior precedent, not a novelty claim. |
| Jeong et al., [Adaptive-RAG, NAACL 2024](https://aclanthology.org/2024.naacl-long.389/), abstract and method overview | A trained smaller LM selects no, single-step, or iterative retrieval using complexity labels derived partly from system outcomes. | Hold retrieval mechanics constant to isolate a binary decision. Our prompted, frozen, same-model gate is not their trained classifier or reproduction. |
| Asai et al., [Self-RAG](https://arxiv.org/abs/2310.11511), abstract | Trains a single LM to retrieve on demand and assess passages/generations with reflection tokens. | Decision-model ownership of retrieval already has close precedent. This study uses existing instruction-model weights and a JSON action interface; it does not reproduce Self-RAG training. |
| De Sabbata et al., [Rational Metareasoning for LLMs, v3](https://arxiv.org/html/2410.05563v3), Sections 2–3 and 7 | Uses a value-of-computation-inspired reward and Expert Iteration to learn selective reasoning. Tool-using agents are a stated future direction. | Count the gate call, repeated input, answer generation, and mechanical search. Fewer searches alone cannot establish lower computation cost. No training or transfer of the paper's empirical claims is assumed. |

Our narrow question is whether this local model's prompted retrieval decision
preserves success while improving measured cost against fixed and observable
policies. A synthetic certificate lookup makes provenance and staleness
inspectable. It can also expose a ceiling for simple code: if the heuristic
resolves development cases, retire that suite instead of manufacturing a weak
baseline or spending held-out evaluation on it. Query writing, evidence selection,
memory writes, training, and multi-step planning remain separate future questions.

The historical MemRL pilot remains closed and inconclusive. Its data are not
development examples, memory records, or test inputs in this experiment.
