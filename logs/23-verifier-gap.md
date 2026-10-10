# Generator–verifier gap probes (cross-family verifier panel on GSM8K; FEVER grounded citations)
**Dates:** Aug 20 (result and log file timestamps; year not stated in the source) · **Status:** smoke tests only; no full evaluation run · **Hardware/scale:** not stated; models served locally through Ollama

## Question / goal
How large is the gap between what one sample from a small LM gets right and what is achievable from a pool of samples (GSM8K)? Can a panel of verifiers from different model families select the correct answer? Does forcing a verbatim quote from the source improve fact verification on FEVER, and does a string check catch invented citations?

## Approach
Two probe scripts plus a labeller, all in one small package (`gvgap`):
- **Verifier-panel probe (`run.py`)**: sample N candidate solutions per GSM8K problem from a generator model. Score each candidate with three verifiers: self-consistency (answer agreement across samples), a judge model from a different family, and an executor (a solver model writes Python, which is run, and its output is compared to the candidate answer). Report pass@1, oracle pass@n, and the routed pick under uniform and fitted verifier weights. Fitted weights are learned on the first 30% of problems and tested on the rest.
- **Grounded citation probe (`cite.py`)**: on FEVER claims, each model answers twice: ungrounded (verdict only) and grounded (verdict plus a verbatim quote). A string match checks whether the quote appears in the document. Half of the documents have the deciding evidence removed on purpose, so a confident verdict there cannot be supported.
- **Labeller (`label.py`)**: produces answers for unlabelled GSM8K-style problems by accepting an answer only when programs from several model families execute to the same value. Gold labels are used only for grading. Not yet run beyond the code.
- Models run through Ollama's local HTTP API. Defaults: generator `qwen3:8b`, judge `phi4`, solver `deepseek-r1:8b`; citation probe `qwen3:8b`, `phi4`, `mistral:7b`.
- Datasets: GSM8K (test split) and FEVER (validation split), loaded with Hugging Face `datasets`.
- Results are cached as JSON so expensive panel scoring runs once.

## Results
Smoke-scale only. No aggregate metric (pass@1, oracle, routed accuracy, or citation rates) has been computed or reported.
- `results/smoke.json` (GSM8K panel probe): 2 problems, 3 candidate answers each, 3 verifier scores per candidate. Gold answers are 18 and 3.
- `results/citesmoke.json`: 12 rows (model, verdict, quote, grounded flag).
- `results/citesmoke2.json`: 16 rows over 8 unique claims and 2 models (`qwen3:8b`, `mistral:7b`); 14 rows had the evidence present and 2 rows had it withheld. Gold labels are 8 rows SUPPORTS and 8 rows REFUTES.
- No `label.py` output file exists.

## Limitations
- Smoke tests only: the number of problems and claims is far too small for any rate to mean anything.
- The panel probe's smoke file does not record which models were used.
- The citation probe's smoke runs used two models; the default third model (`phi4`) does not appear in the smoke output.
- The string-match gate only checks that the quote is present in the document. It does not check that the quote supports the verdict, so a real but irrelevant sentence passes (the `miscited` category in the report).
- The labeller's precision has not been measured, so its usefulness is untested.
- The verifier weights are fitted on a small dev split; with few problems the weights are unstable.
- The project stopped at smoke tests; the full experiments described in the code defaults (50 problems, 8 candidates for `run.py`; 40 claims for `cite.py`; 40 problems for `label.py`) have not been reported.

## Key files
- `run.py`: GSM8K verifier-panel probe; collects candidates and verifier scores, caches them, and reports pass@1, oracle, and routed accuracy.
- `cite.py`: FEVER grounded-citation probe; compares ungrounded and verbatim-quote verdicts, with and without the deciding evidence.
- `label.py`: cross-family labeller; reports coverage, precision, and agreement-tier precision.
- `gvgap/verifiers.py`: the three verifiers (consistency, judge, executor) and the judge prompt.
- `gvgap/router.py`: candidate proposal, panel scoring, and weight fitting.
- `gvgap/grounding.py`: citation generation, the quote-in-document check (`contains`), and the ungrounded judge.
- `gvgap/execute.py` and `gvgap/ground.py`: code-solver execution and the labeller's agreement rule.
- `gvgap/llm.py`: thin Ollama HTTP client.
- `gvgap/data.py`: GSM8K and FEVER loaders.
- `results/smoke.json`, `results/citesmoke.json`, `results/citesmoke2.json`: smoke-test outputs.

## Sources
- GSM8K (grade-school math dataset) and FEVER (fact verification dataset), loaded via Hugging Face `datasets`.
- Ollama, for local model serving.
- Project code: `run.py`, `cite.py`, `label.py`, and the `gvgap` package.
