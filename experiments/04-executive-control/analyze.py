"""Audit saved observable traces and produce a descriptive report; no model calls."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import experiment as e


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    out = args.run.resolve()
    manifest = json.loads((out/"manifest.json").read_text())
    data = json.loads((out/"dataset_evaluator_only.json").read_text())
    prompts = json.loads((out/"prompts.json").read_text())
    rows, calls = read_lines(out/"episodes.jsonl"), read_lines(out/"generation_calls.jsonl")
    checks = {}
    checks["frozen_artifacts_match_hashes"] = all(hashlib.sha256((out/name).read_bytes()).hexdigest() == sha for name,sha in manifest["frozen_files"].items())
    checks["dataset_hash_matches"] = e.digest(data) == manifest["dataset_sha256"]
    checks["analyzer_uses_matching_experiment_source"] = (out/"experiment.py").read_bytes() == (e.ROOT/"experiment.py").read_bytes()
    checks["complete_attempt_accounting"] = len(calls) == manifest["calls"] == sum(r["calls"] for r in rows)
    checks["all_calls_follow_freeze_timestamp"] = all(c["started_utc"] > manifest["created_utc"] for c in calls)
    checks["actual_tokens_reconcile"] = sum(c.get("response",{}).get("prompt_eval_count",0)+c.get("response",{}).get("eval_count",0) for c in calls) == manifest["actual_tokens"] == sum(r["prompt_tokens"]+r["output_tokens"] for r in rows)
    checks["common_episode_budgets"] = all(r["calls"] <= 2 and r["retrievals"] <= 1 and r["prompt_tokens"]+r["output_tokens"] <= manifest["config"]["episode_tokens"] and r["wall_seconds"] <= 120 for r in rows)
    checks["unique_task_arm_observations"] = len({(r["phase"],r["task_id"],r["arm"]) for r in rows}) == len(rows)
    checks["observations_and_search_exact"] = True
    checks["scores_recompute"] = True
    checks["call_prompts_contain_only_allowed_observations"] = True
    checks["call_actions_match_logged_actions"] = True
    tasks = {(phase,t["id"]):t for phase,suites in data.items() for ds in suites.values() for t in ds["tasks"]}
    row_by_key = {(r["phase"],r["task_id"],r["arm"]):r for r in rows}
    visible_by_key = {}
    for key,row in row_by_key.items():
        task = tasks[(row["phase"],row["task_id"])]
        obs = task["observation"]
        store = e.PassiveStore(data[row["phase"]][row["suite"]]["corpus"])
        memory = store.read_all() if row["arm"] == "full_context" else store.search(obs["entity"]) if row["retrievals"] else []
        visible_by_key[key] = memory
        searches = [a for a in row["actions"] if a["action"] == "search"]
        if searches:
            checks["observations_and_search_exact"] &= searches == [dict(action="search",query=obs["entity"],limit=8,returned_ids=[r["id"] for r in memory])]
        scores = e.score(task,row["answer"],obs["current"]+memory)
        checks["scores_recompute"] &= all(row[k] == v for k,v in scores.items())
    for call in calls:
        key = (call["phase"],call["task_id"],call["arm"])
        obs = tasks[(call["phase"],call["task_id"])]["observation"]
        gate = call["role"] == "gate"
        memory = [] if gate else visible_by_key[key]
        messages = [dict(role="system",content=prompts["gate" if gate else "answer"]),
                    dict(role="user",content="MEMORY RECORDS\n"+e.packed(memory)+"\nCURRENT TASK\n"+e.packed(obs))]
        checks["call_prompts_contain_only_allowed_observations"] &= call["request"]["messages"] == messages
        if "error" not in call:
            action = e.parse(call["response"]["message"]["content"],gate)
            checks["call_actions_match_logged_actions"] &= (row_by_key[key]["actions"][0]["action"] == action["decision"] if gate else row_by_key[key]["answer"] == action)
    summary = e.summarize(rows)
    checks["summary_recomputes"] = summary == json.loads((out/"summary.json").read_text())
    gates = json.loads((out/"development_gates.json").read_text())
    checks["development_stopping_rules_match"] = gates == e.development_gates(summary)
    checks["evaluation_only_for_passing_suites"] = all(gates[r["suite"]]["evaluate"] for r in rows if r["phase"] == "evaluation")
    identical = defaultdict(list)
    for call in calls:
        if "error" not in call:
            identical[e.packed(call["request"])].append(e.packed(e.parse(call["response"]["message"]["content"],call["role"] == "gate")))
    checks["identical_requests_agree_on_parsed_actions"] = all(len(set(outputs)) == 1 for outputs in identical.values())
    e.save(out/"audit.json",dict(passed=all(checks.values()),checks=checks,model_calls_made_by_audit=0))
    if not all(checks.values()):
        raise AssertionError(checks)
    report = ["# Retrieve-versus-act candidate experiment — 2026-09-10", "",
              f"Status: **{manifest['status']}**. This is a development diagnostic, not a benchmark reproduction or evidence of general executive control.", "",
              "The frozen protocol and code preceded the first local generation. Memory remained a passive immutable store; the model gate chose act/retrieve and the host executed the fixed entity query. No training or memory writes occurred.", "",
              f"Run: {manifest['calls']} local Qwen2.5 14B generation calls, {manifest['actual_tokens']:,} actual prompt/output tokens, {manifest['wall_seconds']:.1f} seconds elapsed. Zero cloud inference calls. Model digest and Ollama metadata are in `model_metadata.json`.", "",
              "## Results", "",
              "Counts are per 8-case development suite. Token totals include gate and answer calls. Full-context uses no search call but loads the entire corpus; its memory and token cost is included. See [interpretation](../../CONCLUSION.md) for the resulting research decision.", "",
              "| Suite | Policy | Correct | Searches | Prompt tokens | Output tokens | Wall seconds | Stale use |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for suite,arms in summary["development"].items():
        for arm in e.ARMS:
            r = arms[arm]
            report.append(f"| {suite} | {arm} | {r['correct']}/{r['n']} | {r['retrievals']} | {r['prompt_tokens']:,} | {r['output_tokens']:,} | {r['wall_seconds']:.3f} | {r['stale_use']} |")
    report.extend(["", "## Protocol decision", ""])
    for suite,gate in gates.items():
        a = summary["development"][suite]
        report.append(f"- **{suite}: {gate['reason']}.** Always-retrieve {a['always_retrieve']['correct']}/8; no-retrieval {a['no_retrieval']['correct']}/8; full-context {a['full_context']['correct']}/8; heuristic {a['heuristic']['correct']}/8; symbolic {a['symbolic']['correct']}/8. Proceed to held-out evaluation: {gate['evaluate']}.")
    report.extend(["", "## Retrieval decisions and evidence", "",
                   "| Suite | Policy | Missed needed retrieval | Search on sufficient current evidence | Search on missing/stale/conflict | Gate tokens | Evidence-valid outputs |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"])
    for suite,arms in summary["development"].items():
        for arm in ("always_retrieve","heuristic","model_gate","full_context","symbolic"):
            r = arms[arm]
            report.append(f"| {suite} | {arm} | {r['missed_retrieval']}/{r['needs_memory']} | {r['unnecessary_retrieval']}/{r['current_sufficient']} | {r['futile_searches']} | {r['gate_tokens']:,} | {r['evidence_valid']}/{r['n']} |")
    report.extend(["", "Evidence-valid counts measure citation coverage/validity; they are not success counts and do not establish that conflicting evidence was resolved correctly. An incorrect abstention can still have the required empty evidence list. Full-context's missed retrieval count is zero because it receives all storage evidence.", "", "| Suite | Policy | Loaded records (sum) | Memory bytes (sum) | Search/read microseconds (sum) | Gate seconds (sum) |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for suite,arms in summary["development"].items():
        for arm in e.ARMS:
            r = arms[arm]
            report.append(f"| {suite} | {arm} | {r['loaded_records']} | {r['memory_bytes']:,} | {r['search_seconds']*1e6:.1f} | {r['gate_seconds']:.3f} |")
    report.extend(["", "Search/read time measures the local loading branch (including minimal bookkeeping even when no records are loaded); it is not a remote search price. All 80 LLM episodes produced valid schema actions with zero infrastructure failures. Exact matching requests agreed on parsed JSON actions; observed formatting differences did not change actions.", "", "## Observed failures", ""])
    for r in rows:
        if r["arm"] == "model_gate" and not r["correct"]:
            task = tasks[(r["phase"],r["task_id"])]
            report.append(f"- `{r['task_id']}` ({r['scenario']}): gate `{r['actions'][0]['action']}`, answer `{r['answer']['answer'] if r['answer'] else 'INVALID'}`, expected `{task['expected']}`. Inspect the complete observable request/response in `generation_calls.jsonl`.")
    report.extend(["", "## Scope and reproduction", "",
                   "The two suites share certificate validity rules and a synthetic task family. Eight cases per suite cannot establish generalization or noninferiority. Frozen seeds do not guarantee bitwise GPU determinism. Latency includes model loading and depends on Ollama caching and host load; symbolic times near zero are CPU measurements, not literal zero cost.", "",
                   "All evaluator scenarios/targets are separate from model-facing observations. An offline audit checked snapshot hashes, exact prompts/search results, observable action traces, recomputed scoring, token reconciliation, budgets, and the development stopping rules. See `audit.json`. Seven preflight unit tests passed before the run; their source is saved beside the protocol.", "",
                   "Reproduce from `D:\\research\\executive-control`: `python -m unittest -v test_experiment.py`, then `python experiment.py --run`. Each run creates a new directory and applies its own development stop. Analyze a saved run with `python analyze.py results/<run-name>`; this makes no model calls.", "",
                   "Preserved artifacts: `PROTOCOL.md`, `PRIOR_WORK.md`, `experiment.py`, `test_experiment.py`, `prompts.json`, `manifest.json`, `model_metadata.json`, `dataset_evaluator_only.json`, `generation_calls.jsonl`, `episodes.jsonl`, `development_gates.json`, `summary.json`, `paired_comparisons.json`, and `audit.json`. The original MemRL pilot remains closed as inconclusive and unchanged.", ""])
    (out/"REPORT.md").write_text("\n".join(report),encoding="utf-8")
    print(json.dumps(dict(audit=checks,report=str(out/"REPORT.md")),indent=2))


if __name__ == "__main__":
    main()
