"""Offline v2 trace audit and report. Does not make inference calls."""
import argparse
import hashlib
import json
from pathlib import Path

import experiment as v1
import instruction_delivery as d


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    out = parser.parse_args().run.resolve()
    manifest = json.loads((out/"manifest.json").read_text())
    data = json.loads((out/"dataset_evaluator_only.json").read_text())
    plans = json.loads((out/"planned_requests.json").read_text())
    rows,calls = lines(out/"episodes.jsonl"),lines(out/"generation_calls.jsonl")
    summary,gates = json.loads((out/"summary.json").read_text()),json.loads((out/"gates.json").read_text())
    checks = dict(
        frozen_hashes=all(hashlib.sha256((out/name).read_bytes()).hexdigest() == sha for name,sha in manifest["frozen_files"].items()),
        dataset_hash=v1.digest(data) == manifest["dataset_sha256"],
        matching_analysis_sources=all((out/name).read_bytes() == (d.ROOT/name).read_bytes() for name in ("experiment.py","instruction_delivery.py")),
        planned_requests_recompute=plans == d.plan_requests(data),
        attempted_requests_match_frozen_plan=all({"endpoint":c["endpoint"],"request":c["request"]} == plans[c["key"]] for c in calls),
        requests_after_freeze=all(c["started_utc"] > manifest["created_utc"] for c in calls),
        call_accounting=len(calls) == manifest["calls"] == sum(r["calls"] for r in rows),
        actual_token_accounting=manifest["actual_tokens"] == sum(r["prompt_tokens"]+r["output_tokens"] for r in rows) == sum(c.get("response",{}).get("prompt_eval_count",0)+c.get("response",{}).get("eval_count",0) for c in calls),
        common_episode_budgets=all(r["calls"] <= 1 and r["wall_seconds"] <= 60 and r["prompt_tokens"]+r["output_tokens"] <= 16384 for r in rows),
        unique_observations=len({(r["phase"],r["task_id"],r["route"]) for r in rows}) == len(rows),
        summary_recomputes=summary == d.summary_for(rows),
        paired_comparisons_recompute=json.loads((out/"paired_comparisons.json").read_text()) == d.paired(rows),
    )
    indexed = {(r["phase"],r["task_id"],r["route"]):r for r in rows}
    tasks = {(phase,t["id"]):t for phase,suites in data["phases"].items() for dataset in suites.values() for t in dataset["tasks"]}
    controls = {t["id"]:t for t in data["controls"]}
    checks["actions_match_responses"] = True
    checks["scoring_and_search_recompute"] = True
    for call in calls:
        row = indexed[tuple(call["key"].split("|"))]
        if "error" not in call:
            response = call["response"]
            raw = response["response"] if call["endpoint"] == "/api/generate" else response["message"]["content"]
            checks["actions_match_responses"] &= row["action"] == json.loads(raw)
    for row in rows:
        if row["phase"] == "controls":
            expected = "UNKNOWN" if row["route"] == "chat_none" else controls[row["task_id"]]["marker"]
            checks["scoring_and_search_recompute"] &= row["correct"] == bool(row["action"] and row["action"]["answer"] == expected)
        else:
            task = tasks[(row["phase"],row["task_id"])]
            store = v1.PassiveStore(data["phases"][row["phase"]][row["suite"]]["corpus"])
            memory = store.search(task["observation"]["entity"])
            result = v1.score(task,row["action"],task["observation"]["current"]+memory)
            checks["scoring_and_search_recompute"] &= all(row[k] == val for k,val in result.items()) and row["returned_ids"] == [r["id"] for r in memory]
    control = summary["controls"]["nonce"]
    checks["development_gate_recomputes"] = gates["controls_pass"] == (control["chat_none"]["correct"] >= 7 and any(control[r]["correct"] >= 7 for r in d.POSITIVE))
    if "development" in summary:
        checks["candidate_gate_recomputes"] = gates["candidates"] == d.candidates_from(summary)
    if "confirmation" in summary:
        selected,supported = d.selection(summary,gates["candidates"])
        checks["confirmation_gate_recomputes"] = selected == gates["selected_route"] and supported == gates["supported_routes"]
    else:
        checks["confirmation_correctly_skipped"] = not gates.get("candidates") and not any(r["phase"] == "confirmation" for r in rows)
    v1.save(out/"audit.json",dict(passed=all(checks.values()),checks=checks,inference_calls=0))
    if not all(checks.values()):
        raise AssertionError(checks)
    controls_perfect = all(r["correct"] == r["n"] for r in control.values())
    stopped = not gates.get("candidates")
    report = ["# Instruction-delivery diagnostic — 2026-09-10","",
              f"Status: **{manifest['status']}**. Development candidates: {gates.get('candidates',[])}. Confirmation executed: {'confirmation' in summary}.","",
              ("All instruction controls passed. This run does not support the hypothesis that chat system instructions are simply being dropped. It does not establish a universally reliable transport or localize an internal server cause." if controls_perfect else "Not all controls passed; inspect their results before interpreting instruction delivery."),"",
              f"Executed {manifest['calls']} local Qwen2.5 14B generation calls, {manifest['actual_tokens']:,} prompt-plus-output tokens, {manifest['wall_seconds']:.1f} seconds. Infrastructure failures: {manifest['infrastructure_failures']}. No training or cloud inference. The protocol includes five preflight tests; all {len(checks)} offline audit checks passed.","",
              "## Controls","","| Delivery route | Correct marker / expected UNKNOWN | Prompt tokens | Output tokens | Wall seconds |","| --- | ---: | ---: | ---: | ---: |"]
    for route in d.POSITIVE+["chat_none"]:
        r = control[route]
        report.append(f"| {route} | {r['correct']}/{r['n']} | {r['prompt_tokens']} | {r['output_tokens']} | {r['wall_seconds']:.3f} |")
    report.extend(["","The marker was only in the instruction, never in the output schema or no-instruction payload. The negative control expects UNKNOWN. Successful positive controls establish instruction-conditioned behavior on these simple tasks.","","## Evidence-use development","",
                   "Every route received identical current evidence and mechanical search results, including distractors. The semantic instruction and JSON schema were unchanged across routes. Both code alphabet and task identities are fresh, so differences versus the old experiment are not a controlled delivery effect.","",
                   "| Suite | Route | Correct | Evidence-valid | Stale use | Prompt tokens | Cached prompt subset | Output tokens | Wall seconds |","| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for suite,arms in summary.get("development",{}).items():
        for route in d.POSITIVE+["symbolic"]:
            r = arms[route]
            report.append(f"| {suite} | {route} | {r['correct']}/{r['n']} | {r['evidence_valid']}/{r['n']} | {r['stale_use']} | {r['prompt_tokens']:,} | {r['cached_prompt_tokens'] if r['cached_prompt_tokens'] is not None else 'n/a'} | {r['output_tokens']} | {r['wall_seconds']:.6f} |")
    report.extend(["","Cached prompt counts are part of total prompt counts, not additional tokens. Cache reuse differs across routes; latency is descriptive and includes model loading. Evidence-valid citations alone do not prove the right answer or correct conflict handling.","",
                   ("No route reached 9/10 in each suite with zero stale use. Do not select a route merely because it has the highest aggregate score. All 20 new confirmation tasks remain unrun, along with the original study's 48 held-out tasks." if stopped else f"Development candidates: {gates['candidates']}. Confirmation selection: {gates.get('selected_route')}. Consult summary.json for confirmation results."),"",
                   "## Observable failures","",
                   "| Task | Scenario | Route | Actual answer | Expected |","| --- | --- | --- | --- | --- |"])
    for row in rows:
        if row["phase"] == "development" and not row["correct"]:
            task = tasks[(row["phase"],row["task_id"])]
            report.append(f"| {row['task_id']} | {row['scenario']} | {row['route']} | {row['action']['answer'] if row['action'] else 'INVALID'} | {task['expected']} |")
    report.extend(["","Inspect the scenario breakdown above for the evidence judgments that failed. Retrieval intention was held fixed, so these errors cannot be explained by failure to issue a search. The traces do not reveal private reasoning or prove why an instruction was misapplied.","",
                   "## Research decision","",
                   ("Keep the existing chat transport as an unvalidated baseline; neither change the service nor claim an interface repair. The next prerequisite is a fresh, minimal-pair evidence-use diagnostic that isolates entity matching, authority, time validity, conflict handling, and their composition. That should be an answer-policy capability check, not another retrieval-gate contest. Keep the certificate task as a symbolic regression fixture and do not train an executive gate to compensate for its answer-policy failures." if stopped else "Use the confirmation gate before adopting a route. Any later controller comparison requires a separate protocol; evidence-use success alone does not establish a learned routing advantage."),"",
                   "The intended learned executive policy, observations, actions, and learning target are specified in [MODEL.md](../../MODEL.md). This run trains no executive model and supplies no evidence of an advantage over simple code.","",
                   "## Reproduction and preserved evidence","",
                   "From `D:\\research\\executive-control`: `python -m unittest -v test_instruction_delivery.py`, then `python instruction_delivery.py --run`. Analyze a saved directory with `python analyze_delivery.py results/<run-name>`. The audit makes no inference calls.","",
                   "The run stores the protocol, source snapshots, dataset, every planned request, every attempted request/response, model template and version, summaries, paired comparisons, gate decisions, and audit. All preceding result directories and MemRL files are preserved. Small samples and shared synthetic templates limit generalization; paired comparisons are descriptive.",""])
    (out/"REPORT.md").write_text("\n".join(report),encoding="utf-8")
    print(json.dumps(dict(audit=checks,report=str(out/"REPORT.md")),indent=2))


if __name__ == "__main__":
    main()
