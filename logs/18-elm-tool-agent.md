# ELM: LoRA-tuning small open models into a local tool-using research agent
**Dates:** Sep 21, 2026 (all dated records and repository commits) · **Status:** Ongoing; the default launcher uses untrained Qwen3.5-2B, and no 2B adapter exists yet · **Hardware/scale:** One local RTX 4080 SUPER (16 GB), about 32 GB RAM; no cloud GPU · **Code:** https://github.com/LamShiuChing/Language-Model-Optimization

## Question
Can a small pretrained model, adapted with LoRA or full fine-tuning on tool-action traces and paired with web search, page reading, and local tools, work as a useful local research agent? A second question is how much of any success comes from the model and how much comes from host-side checks.

## Approach
- **Base models:** SmolLM2-135M-Instruct and SmolLM2-360M-Instruct (first pilot and stages 2–5, full-model training); Qwen3-0.6B (frozen 600M history, rank-16 LoRA); Qwen3.5-2B, revision `15852e8…`, no adapter (current default in `chat.cmd`).
- **Adapter parameters:** Qwen3-0.6B LoRA (r=16, alpha=32, on q/k/v/o/gate/up/down projections) trained 10,092,544 adapter parameters over a frozen 596,049,920-parameter base (606,142,464 total with base).
- **Training data:**
  - Fictional grouped scenario curriculum: 128 training groups, eight variants each (original facts, changed facts, missing evidence, distractors, transient read error, injected instructions, long page, browser fallback). Dev and test groups are split by group.
  - Stage 4 and later: answers placed in non-final chunks, plus multiple search-result positions.
  - Natural passages from the SQuAD 2.0 training file, split by article; `natural-v1` has 2,864 training episodes, including 940 unanswerable and 480 replay episodes. The official dev/test set was not used.
  - Agent-flow continuation: 894 action examples, 240 updates.
  - Conversation continuation: 384 direct conversation/reasoning examples and 299 tool-replay examples, 256 updates.
- **Training updates by stage:** 135M: 320 (stage 2), 640 (stage 3), 640 (stage 4), 1,000 (stage 5). 360M: 320 (stages 3 and 4). Qwen3-0.6B LoRA: 320 (natural curriculum).
- **Tools:** Search (Bing RSS and Wikipedia, merged and deduplicated); direct HTTP fetch, with Chromium fallback; chunked `read` with `next_offset`; calculator; volume statistics; list/read/write files and shell commands. Writes and shell commands require host approval. Shell runs with user privileges, not in an OS sandbox.
- **Host controller:** Plan requirements, byte-normalized quoted evidence from read sources, SHA-256 freshness for execution checks, duplicate-search rejection, at most 3 repairs, and at most 2 invalid-action responses. Only tool observations update state.
- **Benchmarks:** Synthetic dev groups (2 groups, then 8 groups); natural-passage screens; live web probes; a 22-case dev benchmark (600M); a 6-case lookup comparison (2B, offline); a 5-turn smoke test; 91 software tests.

## Results
| Stage / model | Test | Result | Verdict |
|---|---|---|---|
| First pilot, original 135M | Synthetic dev, 2 groups × 8 variants | 0/16 success; 96 invalid actions | Fail (baseline) |
| First pilot, adapted 135M (80 updates) | Same synthetic dev | 16/16 success; 0 invalid actions | Pass (synthetic only) |
| First pilot, original 360M | Same synthetic dev | 0/16 success; 96 invalid actions | Fail (baseline) |
| First pilot, 135M-class live run | 4 live development questions | 0/4 | Fail |
| Stage 2, 135M (320 updates) | Synthetic dev, 8 groups | 32/72; live 0/4 | Partial / Fail (live) |
| Stage 3, 135M (640 updates) | Synthetic dev, 8 groups | 63/72; live 0/4 | Partial / Fail (live) |
| Stage 3, 360M (320 updates) | Synthetic dev | Not evaluated; live 0/4 | Fail (live) |
| Stage 4, 135M (640 updates) | Natural passage baseline | 4/16; live 0/4 | Fail |
| Stage 4, 360M (320 updates) | Natural passage baseline | 2/16; live 0/4 | Fail |
| Stage 5, 135M (1,000 updates) | Natural passage screen | 5/16 (1/8 answerable, 4/8 unanswerable); live 0/4 | Fail |
| Qwen3-0.6B, unadapted reference | Live probes | 0/4 strict; 0/4 with prompt/fence diagnostic | Fail |
| Qwen3-0.6B + conversation LoRA (frozen 600M baseline) | 22-case dev benchmark | 19/22: conversation 6/6, arithmetic 3/4, logic 3/4, controlled evidence 4/4, live web 3/4 | Partial pass (selected suite, not an accuracy estimate) |
| Qwen3-0.6B + conversation LoRA | 5-turn smoke test | Passed greeting, prior-turn recall, categorical inference, sourced live Rust lookup; failed arithmetic (expected 33) | Partial |
| Qwen3-0.6B + agent-flow LoRA (240 updates) | Live arithmetic `2050202 + 9299` | Returned 2059501 | Pass |
| Qwen3-0.6B + agent-flow LoRA | Live D-drive usage | Used 87.7% correctly; total capacity garbled | Partial |
| Qwen3-0.6B + agent-flow LoRA | Research and implement classic Snake | No game created; answered about an irrelevant result | Fail |
| Qwen3-0.6B + agent-flow LoRA | Kilometers-to-miles converter (write, run) | NameError on execution; final answer falsely claimed success | Fail |
| Qwen3.5-2B, no adapter (stronger-model check) | Exact arithmetic | Correct: 2059501 | Pass |
| Qwen3.5-2B | D-drive usage | Accurate summary in GB after one invalid format | Pass with format error |
| Qwen3.5-2B | Greeting (first-action probe) | Friendly text, not required JSON | Fail (format) |
| Qwen3.5-2B | Research and implement Snake | Searched, then 4 invalid actions; context exhausted | Fail |
| Qwen3.5-2B | Inspection of generated HTML (supervisor-extracted, not model-run) | 7 of 7 basic headless-browser checks passed | Pass (not autonomous completion) |
| Qwen3.5-2B + task controller | Live arithmetic `2050202 + 9299` | Answered 2,059,501; 1 calculator call; 4.99 s | Pass |
| Qwen3.5-2B + task controller | Live Tetris SRS research (rules only) | `verification_failed`; 3 searches, 0 reads; 32.61 s; unread citations rejected | Fail |
| Qwen3.5-2B, lookup comparison, JSON interface | 6 synthetic dev cases, one run | 0/6 strict; 12 invalid actions; manual review 0/6 correct sourced completions | Fail |
| Qwen3.5-2B, lookup comparison, native calls | Same 6 cases | 0/6 strict; 1 invalid action; manual review 3/6 | Fail |
| Qwen3.5-2B, native calls + BM25 | Same 6 cases | 2/6 strict; 0 invalid actions; manual review 4/6 | Partial (one run) |
| Software suite | `pytest` | 91 passed | Pass (software only) |

## What it showed
- Small-model adaptation taught the action format on synthetic tasks: the first pilot's 135M model went from 0/16 to 16/16 with 0 invalid actions. This did not carry over to live web research. Every live probe in the 135M/360M stages (0/4 each) and the unadapted Qwen3-0.6B reference (0/4 strict) failed. Later Qwen3-0.6B conversation runs did pass some live web tasks (3/4 in the 22-case suite).
- The 600M conversational baseline passed 19/22 on a small selected dev suite, including 3/4 live web tasks. It still failed a simple arithmetic word problem, and the agent-flow adapter falsely reported a successful run after an execution error.
- Qwen3.5-2B, used without training, did better on arithmetic and generated HTML than the 600M models, but its autonomous Snake workflow failed. The HTML check used supervisor extraction and is not evidence of autonomous completion.
- Host checks blocked unverified answers (such as unread citations and duplicate searches), but they did not produce a successful live research answer. Blocking an unsupported answer is not the same as the model doing the task. In the Tetris probe, the model never read a page.
- In the 6-case lookup comparison, a local BM25 index raised strict success from 0/6 to 2/6, and manual review from 3/6 to 4/6. The comparison was one run and used a deliberately constructed fixture. The records attribute most remaining failures to retrieval quality and output format.

## Limitations
- All evaluations are small, selected development sets, usually with one seed. Live probes were used repeatedly during development, so they are not unseen benchmarks. The final test splits were never used.
- No 2B adapter has been trained. The default chat model is the untrained Qwen3.5-2B.
- The 600M results reflect a single 22-case dev run and do not estimate general accuracy.
- Energy was not measured. The inherited pretraining cost of each base model is unknown. Local adaptation time excludes upstream pretraining.
- The host checks confirm provenance, freshness, and execution. They cannot verify that a quoted passage is relevant or true, or that a plan covers the user's requirements.
- Shell approval is not an OS sandbox.
- Search providers can change or challenge requests. DuckDuckGo returned a challenge, and Bing RSS worked during these runs.
- Browser fallback only renders pages. It does not click, fill forms, scroll, or use logins.
- Cleanup removed 17 files (15.832 GiB), including optimizer states and the obsolete 135M/360M full-model weights. Historical runs can no longer resume directly.
- The four first-pilot live probes are not tied to a named model in the records, so that row is attributed to the first pilot.
- The SQuAD-derived data is a transformed training-split curriculum, not an official SQuAD benchmark.

## Key files
- `README.md`: project overview, run instructions, and dated results summary.
- `docs/STATUS.md`: dated results table, stage history, and known gaps.
- `docs/THESIS.md`, `docs/HANDOFF.md`: working thesis, current runtime identity, and next steps.
- `docs/Project_Review_2026-09-21.md`: prior-work positioning and proposed phases.
- `configs/agent-2b.json`: current default configuration (Qwen3.5-2B, budgets, system prompt).
- `configs/qwen-600m-lora.json`, `configs/conversation-train.json`: Qwen3-0.6B LoRA settings (r=16).
- `configs/tool-stage2.json` through `configs/tool-stage5-360m.json`: 135M/360M stage configurations.
- `baseline.json`: frozen identity of the 600M conversational baseline.
- `chat.cmd`: the only chat launcher.
- `elm/protocol.py`, `elm/agent.py`: action schema and bounded execution loop.
- `elm/taskstate.py`: host plan, evidence, freshness, and repair checks.
- `elm/webtools.py`: search, direct fetch, Chromium fallback, chunked reading.
- `elm/localtools.py`, `elm/computer.py`: calculator, volume statistics, file tools, and approved shell execution.
- `elm/data.py`: synthetic grouped curriculum.
- `elm/natural_data.py`: SQuAD-derived tool trajectories, split by article.
- `elm/train.py`: supervised training (loss on action tokens only; LoRA through PEFT).
- `elm/evaluate.py`: episode evaluation with grouped uncertainty.
- `elm/retrieval.py`: SQLite FTS5 BM25 index, used in the offline comparison.
- `elm/native_tools.py`: parser for the Qwen3.5 native tool-call format.
- `scripts/check_task_controller.py`: live controller probe.
- `reports/Stronger_Model_Check_2026-09-21.md`, `reports/Task_Controller_Check_2026-09-21.md`, `reports/Lookup_Comparison_2026-09-21.md`, `reports/Agent_Flow_Pilot_2026-09-21.md`: dated experiment records.
- `tests/`: software test suite.

## Sources
- `README.md`
- `docs/STATUS.md`
- `docs/HANDOFF.md`
- `docs/THESIS.md`
- `docs/Project_Review_2026-09-21.md`
- `docs/research-direction.md`
- `docs/planner-reuse-audit-2026-09-21.md`
- `reports/Stronger_Model_Check_2026-09-21.md`
- `reports/Task_Controller_Check_2026-09-21.md`
- `reports/Lookup_Comparison_2026-09-21.md`
- `reports/Agent_Flow_Pilot_2026-09-21.md`
- `baseline.json`
- `configs/agent-2b.json`, `configs/qwen-600m-lora.json`, `configs/conversation-train.json`, `configs/tool-pilot.json`
- Not read: `reports/ELM_Benchmark_Record_2026-09-21.docx` (the .docx file was excluded by the task instructions). The 22-case results come from `docs/STATUS.md`, which cites that record.
