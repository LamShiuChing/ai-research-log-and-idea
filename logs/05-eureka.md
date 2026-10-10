# eureka: Learning by Accretion (knowledge, source trust and skills outside a frozen core)
**Dates:** 2026-08-31 (all lab runs, unified runs and docs) · **Status:** ongoing · **Hardware/scale:** one RTX 4080 SUPER (16 GB); lab models 10.6M to 89.5M parameters; SmolLM3-3B for the transfer test and the chat system
Paper: see paper/eureka.pdf
PAPER-WORTHY: already has paper

## Question
Can knowledge, source trust and new skills be added to a frozen language-model core without retraining it and without erasing what it already knows? The project tests this in two parts: toy-scale controlled experiments (`lab/`) and a local chat system on SmolLM3-3B (`unified/`) that carries the parts that work without training. The question matters because it bears on whether a deployed assistant can keep learning without catastrophic forgetting or full retraining.

## Approach
- **Models:** nanoGPT fork (`lab/gpt/`), mostly 10.64M-parameter models (6 layers, 384 dim); EP3 and EP3-natural at 10.65M; EPX at 89.46M (GPT-2-small dimensions, word-level tokens); E11 adapters on GPT-2 124M; EP5 and the `unified/` chat on SmolLM3-3B.
- **Data:** synthetic fact worlds with invented entities and known ground truth (colors, capitals, continents, sizes); EP3 and EP3-natural use 113 real countries; EP5 uses 500 invented countries; E7 to E10 use arithmetic checked by an executable oracle; E11 uses databricks-dolly-15k (8,657 pairs).
- **Method:** a per-token source-id embedding (provenance channel, `n_sources`); kNN output interpolation as an external memory; zero-init bottleneck adapters and LoRA; a linear router on the frozen base's hidden state; verified self-distillation ("sleep"); reversible LoRA merge; a verification-grade store (G0 oracle, G1 anchored source, G2 self-consistent, G3u user-taught, G4 declined).
- **Controls and baselines:** every experiment has a named control (no-tag control, naive fine-tune, replay, bare base, prompt-only arm A0, baseline model trained without the change).
- **Seeds:** single seed for all lab results (stated in the top-level README). EP5 is the first pre-registered experiment and also single seed.
- **Pre-registered bars:** EP5 criteria: transfer is "real" if A2 minus A1 is at least 20 points on heldout truth for both topics with lie rate at most 10%, and the regression check drops at most 2 points. "Not a differentiator" if A0 is at least A2 on heldout truth. E8 uses a self-set acceptance gate of 50/100.

## Results
| Test | Result | Baseline / control | Verdict |
| --- | --- | --- | --- |
| E2 abstention (heldout entities) | fabricated 30/100, abstained 70/100; known facts correct 100/100 | baseline: fabricated 100/100 | positive, toy scale |
| E2b contested evidence (heldout contested) | 58 disputed, 31 majority color, 9 minority, 2 other | false-alarm floor 32 (consistent heldout falsely disputed) | weak positive |
| E3 confidence probe, AUC | 0.458 (probe); 0.479 (LM max softmax) | 0/100 fabrications blocked at gates 0.5 and 0.9 | null |
| E3b joint gradients, AUC | 0.521 | saturated at 1.000 | null |
| E3c injected boundary states, AUC | 0.500 (detached); 0.487 (joint) | 37/50 one-shot lines were memorized | null |
| EP provenance (heldout entities) | tagged: truth 97, lie 1, other 2 | untagged control: truth 29, lie 56, other 15 | positive, toy scale |
| EP2 per-topic trust (heldout) | color truth 85, lie 5; size truth 84, lie 8 | scalar trust would show one topic near 0 | positive |
| EP3 natural English (heldout, n=20) | capital truth 4, lie 0; continent truth 10, lie 3 | trust holds; recall fails at 10M char level | mixed |
| EP4a anchor poisoning (heldout truth / lie) | 0%: 100/0; 10%: 92/4; 25%: 69/20; 50%: 46/47 | linear decline, no cliff | graceful degradation |
| EP4b v1, coverage hole (size, heldout) | truth 0, lie 0, other 100 | color sanity: truth 91, lie 2 | inconclusive (fails closed; not a trust measurement) |
| EP4b v2, format anchored (size, heldout) | truth 5, lie 82, other 13 | format entities 20/20; color heldout truth 95 | positive (fails open) |
| EPX scale, 89.46M (heldout, n=200) | capital 191 truth / 0 lie / 9 other; continent 199 / 1 / 0 | EP3 at 10M char: 4/20 | positive, **confounded** (params, tokenization, iterations changed together) |
| E6 kNN memory (heldout capitals, of 20) | λ=0.75: 20/20; λ=0.0: 4/20 | memory off | positive |
| E5 continual learning (task A / task B, of 100) | naive 15 / 100; replay 100 / 100; kNN store 100 / 100 | base: A 100, B 5 | positive (facts only) |
| E7 skills, v2 (task A facts / task B addition) | naive 0 / 100; adapter alone 0 / 63; adapter + routing 100 / 63 | replay 100 / 100 | positive for routing |
| E7b learned router | 150/150 router accuracy; colors 100/100; addition 63/100; subtraction 97/100 | hand-rule routing | positive |
| E8 self-learning loop | flagged 53 of 200 (all 50 multiplications, 3 false alarms); first adapter 36/100 rejected; escalated adapter 73/100; final routed multiplication 74/100 | addition 63/100, subtraction 100/100 | positive, requires oracle |
| E9 sleep (bare consolidated base) | colors 100; addition 92; subtraction 92; multiplication 81 | modules before: 100, 63, 100, 74 | positive; no unfiltered control |
| E10 curiosity | division 0/30 before; 100/100 on fresh validation probes after one cycle; cycle 2: 30/30 (adapter) | plus 29/30, minus 25/30, times 21/30 before the cycle (no control arm) | positive, grammar is hand-given |
| E11 chat adapter (GPT-2 124M) | 2.37M-parameter adapter; loss 6.24 to 2.48; answers "Paris" to capital question | raw GPT-2 rambles | qualitative only; no scored eval |
| E12a self-consistency gate | memorized: 49/51 correct accepted; fabrication-prone: 1/5 accepted, 15/20 rejected | threshold 8/8 | negative: 5 stable fabrications passed |
| E12b anchored source trust (clean / compromised) | 98/113 (clean); 122/171 (compromised) | fair majority: 76/113; 68/171 | positive |
| E12c graded calibration | G0 30 answered at 100%; G3u 30 at 100%; G2 49 at 78%; G4 11 declined (would have been 27%) | accuracy monotone in grade | positive; G0/G3u scored against themselves |
| E13 reversible LoRA merge | heldout addition: live 73; merged 73; unmerged 0 (A restored to 100) | unrouted merge erases skill A (0) | positive for exact merge; negative for always-on merge |
| EP5 transfer to SmolLM3-3B, heldout capital (truth/lie/other) | A0 prompt-only 73/27/0; A1 LoRA 49/51/0; A2 channel 47/49/4; A3 prefix tag 47/52/1 | A2 minus A1 = -2 (bar: +20) | negative: transfer not shown |
| EP5 transfer, heldout region (truth/lie/other) | A0 63/30/7; A1 53/32/15; A2 49/39/12; A3 48/40/12 | A2 minus A1 = -4 (bar: +20) | negative |
| EP5 A2b (source table at 25x learning rate), heldout capital / region | 54/39/7 and 44/38/18 | A1: 49 and 53 truth | negative (below bar; from `results.jsonl` only, not discussed in README) |
| EP5 regression check, real capitals (of 40) | base 40; A1 29; A2 32; A3 27; A2b 29 | limit: drop of at most 2 | negative: fine-tuning cost real-fact recall |
| unified v2, memory and probes | memory 11/11 (as cited in v3 section); single-op math verified; chains 1/4 | v3 comparison | partial; README lists open failures |
| unified v3, tool-driven loop (38 probes, 3 runs) | three distinct failure modes (tagless calls; six-turn lecture loop; instruction parroting) | v2 rules | negative at 3B |

## What it showed
- At toy scale, a source-id channel in the input let models learn which sources to trust from a small anchored set, with no reliability labels (EP 97 vs 29; EP2 85/84 per topic). Trust degraded roughly linearly as the anchor was corrupted (EP4a: truth 100, 92, 69, 46 at 0, 10, 25, 50% poison).
- Trust also leaked into topics the anchor did not cover, but only once the answer format was taught (EP4b v2: 80 to 82% lies adopted in the unanchored topic). Without the format, the model produced off-topic output (EP4b v1), so that run says nothing about trust.
- Facts can be added instantly through an external store (E5, E6), and new skills can be routed to adapters so old skills survive (E7, E7b). Frozen weights protect parameters, not behavior, so routing was required.
- Two internal self-checks failed (E3 series, E12a): the model's own confidence did not separate known from unknown answers, and self-consistency let stable fabrications through. The mechanisms that worked drew on an external check (oracle, anchored source, user).
- On SmolLM3-3B, the provenance channel did not transfer. Prompt-only (A0) beat every trained arm on heldout truth, and fine-tuning reduced real-capital recall from 40 to 27-32 of 40. The unified v3 tool loop also failed at 3B. The README's conclusion is that at 3B, model-driven control loses to the rule-based v2 harness on every category it tested.

## Limitations
- Single seed for almost all results; the README says differences under about 15 points are noise. EP5 was not repeated across data seeds.
- Toy and synthetic scale: most models are 10.6M parameters, with invented entities and fixed claim ratios (for example 9 vs 6 votes). The only run with real entities is EP3-natural (113 real countries, character-level, recall capped), plus EPX's 12 real entities.
- EPX (89.46M) changed parameters, tokenization and iteration count together, so its scale claim is confounded (stated in the README).
- Several controls are missing or self-referential: E9 has no unfiltered control; E12c scores G0 and G3u against themselves; E8 uses an oracle the loop depends on.
- Small probe sets for the unified system: 9, 27 and 38 probes, one sampled draw each in the first memory run. Pass counts for the 27-probe extraction run are not recorded in the README.
- The E11 chat result is qualitative (three questions), with no scored evaluation.
- The EP5 A2b arm is in `lab/ep5-smollm3/results.jsonl` but not discussed in its README.
- `lab/README.md` has an empty results log; the per-experiment READMEs hold the results.
- The unified v3 training run (`unified/train_v3.log`) shows 1500 conversations; the README says the result of verified-trace training is pending.

## Key files
- `README.md`: project overview, lab findings table and the unified system's measured results.
- `PLAN.md`: working rules (nothing enters memory without a source; nothing enters weights without an outside check), keep/leave decisions and the build order.
- `lab/gpt/model.py`: nanoGPT fork; provenance embedding `wse` enabled by `n_sources` (around lines 133, 152-203).
- `lab/e2-abstention/`, `lab/e2b-contested/`, `lab/e3-confidence/`: abstention and confidence experiments.
- `lab/ep-provenance/`, `lab/ep2-topic-trust/`, `lab/ep4a-anchor/`, `lab/ep4b-coverage/`: source-trust experiments and their controls.
- `lab/epx-scale/`: 89.46M-parameter run.
- `lab/e6-memory/knn.py`: kNN output interpolation over a datastore.
- `lab/e7-skills/adapter_train.py`, `lab/e7-skills/e7b_router.py`: adapter training and the learned router.
- `lab/e8-self/controller.py`: the self-learning loop with the oracle gate.
- `lab/e12-verify/e12b_trust_store.py`: anchored source-trust store.
- `lab/e13-lora/lora_train.py`, `lab/e13-lora/merge_eval.py`: LoRA training and reversible merge.
- `lab/ep5-smollm3/prepare.py`, `train.py`, `eval.py`, `results.jsonl`: SmolLM3-3B transfer test and its pre-registered criteria.
- `unified/store.py`: graded provenance memory (G0 to G4 grades).
- `unified/serve_v2.py` (port 8344), `unified/extract.py`: best measured chat system (rules plus calculator audit).
- `unified/mind_v3.py`, `unified/serve_v3.py` (port 8346): model-driven tool loop (negative at 3B).
- `unified/traces.py`, `unified/train_v3.py`: verified-trace collection and training harness.

## Sources
- `README.md`
- `PLAN.md`
- `lab/README.md`
- `lab/e2-abstention/README.md`
- `lab/e2b-contested/README.md`
- `lab/e3-confidence/README.md`
- `lab/e5-continual/README.md`
- `lab/e6-memory/README.md`
- `lab/e7-skills/README.md`
- `lab/e8-self/README.md`
- `lab/e9-sleep/README.md`
- `lab/e10-curiosity/README.md`
- `lab/e11-chat/README.md`
- `lab/e12-verify/README.md`
- `lab/e13-lora/README.md`
- `lab/ep-provenance/README.md`
- `lab/ep2-topic-trust/README.md`
- `lab/ep3-natural/README.md`
- `lab/ep4a-anchor/README.md`
- `lab/ep4b-coverage/README.md`
- `lab/ep5-smollm3/README.md`
- `lab/ep5-smollm3/results.jsonl`
- `lab/epx-scale/README.md`
- `lab/gpt/README.md`
- `lab/gpt/model.py` (provenance embedding location only)
- `unified/README.md`
- `unified/train_v3.log`
