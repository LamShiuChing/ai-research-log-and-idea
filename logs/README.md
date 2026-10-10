# Project logs & papers

One log per AI project: question, approach, results (numbers copied from each project's own reports, with
sources), what it showed, and limitations. Projects with a clear hypothesis, controlled comparisons and a
conclusion also have a short workshop-style paper. These are write-ups of my own experiments, not
peer-reviewed publications.

| # | Project | Paper |
|---|---|---|
| 01 | [BabyBrain byte lab](01-babybrain-byte-lab.md) | [PDF](papers/01-babybrain-byte-lab.pdf) |
| 02 | [Javis (phases 2–5): learning unseen rules, then a compression + program-library machine](02-javis.md) | [PDF](papers/02-javis.pdf) |
| 03 | [Generative Learning Transformer (fact memory slots on a frozen backbone)](03-generative-learning-transformer.md) | [PDF](papers/03-generative-learning-transformer.pdf) |
| 04 | [Generational consolidation agent + CAM hypervector program learner](04-consolidation-agent-cam.md) | [PDF](papers/04-consolidation-agent-cam.pdf) |
| 05 | [eureka: Learning by Accretion (knowledge, source trust and skills outside a frozen core)](05-eureka.md) | [PDF](../experiments/05-eureka-provenance/paper/eureka.pdf) (original) |
| 06 | [Continual learning falsification testbed (re-testing Dohare et al., Nature 2024, loss of plasticity)](06-plasticity-claims.md) | [PDF](papers/06-plasticity-claims.pdf) |
| 07 | [Self-improvement loops: when does a model improve itself vs drift?](07-self-improvement-loops.md) | [PDF](papers/07-self-improvement-loops.pdf) |
| 08 | [OPAL: DreamCoder-style program induction with a learned library and LLM proposers](08-opal.md) | [PDF](papers/08-opal.pdf) |
| 09 | [Small adaptive systems: can compact learners improve through experience under fixed limits?](09-small-adaptive-systems.md) | [PDF](papers/09-small-adaptive-systems.pdf) |
| 10 | [Executive control of retrieval](10-executive-control.md) | [PDF](papers/10-executive-control.pdf) |
| 11 | [MemRL reproduction (learned memory selection) on a local toy task](11-memrl-reproduction.md) |  |
| 12 | [Fruit-fly connectome networks (flynet) and structured sparse recurrence (SRCN)](12-fruit-fly-connectome.md) | [PDF](papers/12-fruit-fly-connectome.pdf) |
| 13 | [Verified multi-step deduction in small transformers](13-verified-deduction.md) | [PDF](papers/13-verified-deduction.pdf) |
| 14 | [LM optimization: how small GPTs (5M–85M) learn reasoning — 31 findings](14-lm-optimization.md) | [PDF](papers/14-lm-optimization.pdf) |
| 15 | [Compact continual-learning prototypes (frozen MiniLM + online heads on BANKING77 / CLINC150)](15-compact-continual-learners.md) | [PDF](papers/15-compact-continual-learners.pdf) |
| 16 | [Decision model (LCM): one trained scorer for answers and memory/tool actions](16-decision-model-lcm.md) | [PDF](papers/16-decision-model-lcm.pdf) |
| 17 | [Generative proposer: can a CVAE design Boolean circuits for unseen truth tables?](17-generative-proposer.md) |  |
| 18 | [ELM: LoRA-tuning small open models into a local tool-using research agent](18-elm-tool-agent.md) |  |
| 19 | [ARC-AGI-3 interactive-game agents](19-arc-agi-3.md) |  |
| 20 | [Brain-like continual learner: fact memory, slots, sleep on its own dreams, continual GPT](20-brain-like-continual-learner.md) | [PDF](papers/20-brain-like-continual-learner.pdf) |
| 21 | [Continual learning research: concept library → ARC-AGI-3 experience cortex → locked language blocks](21-locked-blocks.md) |  |
| 22 | [Test-time memory chat app (Titans NeuralMemory on a frozen LM)](22-ttt-memory-app.md) |  |
| 23 | [Generator–verifier gap probes (cross-family verifier panel on GSM8K; FEVER grounded citations)](23-verifier-gap.md) |  |
| 24 | [Moji: real-time voice English practice partner](24-moji-voice-tutor.md) |  |

Paper sources are in `papers/*.md`; `python tools/md_to_pdf.py` rebuilds the PDFs.
