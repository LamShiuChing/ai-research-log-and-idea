"""Protocol v2: test instruction transport before interpreting executive policy."""
import argparse
import copy
import hashlib
import json
import math
import platform
import random
import shutil
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import experiment as v1

ROOT = Path(__file__).resolve().parent
POSITIVE = ["chat_system", "chat_user", "raw_system"]
CONFIG = dict(model=v1.MODEL, digest=v1.DIGEST, seed=42, temperature=0,
              num_ctx=16384, num_predict=96, max_calls=152, max_seconds=900,
              max_tokens=2490368, episode_seconds=60)
NONCE_PAYLOAD = 'Return the marker required by the governing instructions. If no marker was specified, return UNKNOWN. Respond with JSON containing only answer.'
NONCE_SCHEMA = {"type":"object","properties":{"answer":{"type":"string"}},
                "required":["answer"],"additionalProperties":False}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def raw_prompt(instruction, payload):
    return f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{payload}<|im_end|>\n<|im_start|>assistant\n"


def request_for(route, instruction, payload, schema):
    request = dict(model=CONFIG["model"],stream=False,format=schema,keep_alive="5m",
                   options={k:CONFIG[k] for k in ("seed","temperature","num_ctx","num_predict")})
    if route == "raw_system":
        request.update(raw=True,prompt=raw_prompt(instruction,payload))
        endpoint = "/api/generate"
        text = request["prompt"]
    else:
        if route == "chat_system":
            messages = [dict(role="system",content=instruction),dict(role="user",content=payload)]
        elif route == "chat_user":
            messages = [dict(role="user",content=instruction+"\n\n"+payload)]
        elif route == "chat_none":
            messages = [dict(role="user",content=payload)]
        else:
            raise ValueError("Unknown delivery route")
        request["messages"] = messages
        endpoint = "/api/chat"
        text = "\n".join(m["content"] for m in messages)
    if len(text.encode())+512+CONFIG["num_predict"] > CONFIG["num_ctx"]:
        raise ValueError("Pre-dispatch context bound exceeded")
    return dict(endpoint=endpoint,request=request)


def make_data():
    controls = []
    rng = random.Random(101001)
    for i in range(8):
        marker = f"M{rng.getrandbits(48):012X}"
        controls.append(dict(id=f"nonce/{i}",marker=marker,
                             instruction=f'The required marker is {marker}. Return JSON with answer equal to that exact marker.'))
    phases = {}
    for phase,seed in (("development",101101),("confirmation",101201)):
        rng = random.Random(seed)
        suites = {}
        for suite,fields in (("access",["access"]),("dispatch",["route","seal"])):
            tasks,corpus = [],[]
            for scenario in ("current","recall","expired_current","unofficial_current","future_current",
                             "wrong_entity_current","missing","stale_only","conflict","current_stale"):
                entity = f"v2-{rng.getrandbits(64):016x}"
                current,memory,values = [],[],[]
                def record(field, value=None, **changes):
                    r = dict(id=f"v2r-{rng.getrandbits(64):016x}",entity=entity,field=field,
                             value=value or f"C{rng.getrandbits(24):06X}",source="registry",
                             issued="2026-01-01",expires="2026-12-31")
                    r.update(changes)
                    return r
                for j,field in enumerate(fields):
                    good = record(field)
                    values.append(good["value"])
                    if scenario in ("current","current_stale"):
                        current.append(good)
                        if scenario == "current_stale":
                            memory.append(record(field,expires="2026-02-01"))
                    elif scenario == "recall":
                        (current if suite == "dispatch" and j == 0 else memory).append(good)
                    elif scenario in ("expired_current","unofficial_current","future_current","wrong_entity_current"):
                        changes = {"expired_current":dict(expires="2026-02-01"),
                                   "unofficial_current":dict(source="chat"),
                                   "future_current":dict(issued="2026-10-01"),
                                   "wrong_entity_current":dict(entity="other-"+entity)}[scenario]
                        current.append(record(field,**changes))
                        memory.append(good)
                    elif scenario == "stale_only":
                        memory.append(record(field,expires="2026-02-01"))
                    elif scenario == "conflict":
                        memory.extend([good,record(field)])
                if scenario not in ("missing","stale_only"):
                    memory.append(record(fields[0],source="chat"))
                rng.shuffle(current)
                tasks.append(dict(id=f"v2/{seed}/{suite}/{len(tasks)}",suite=suite,scenario=scenario,
                                  observation=dict(entity=entity,fields=fields[:],as_of="2026-09-10",current=current),
                                  expected="ABSTAIN" if scenario in ("missing","stale_only","conflict") else "/".join(values)))
                corpus.extend(memory)
            rng.shuffle(tasks)
            rng.shuffle(corpus)
            suites[suite] = dict(tasks=tasks,corpus=corpus)
        phases[phase] = suites
    return dict(controls=controls,phases=phases)


def payload_for(task,memory):
    return "MEMORY RECORDS\n"+v1.packed(memory)+"\nCURRENT TASK\n"+v1.packed(task["observation"])


def plan_requests(data):
    plans = {}
    for task in data["controls"]:
        for route in POSITIVE+["chat_none"]:
            plans[f"controls|{task['id']}|{route}"] = request_for(route,task["instruction"],NONCE_PAYLOAD,NONCE_SCHEMA)
    for phase,suites in data["phases"].items():
        for dataset in suites.values():
            store = v1.PassiveStore(dataset["corpus"])
            for task in dataset["tasks"]:
                memory = store.search(task["observation"]["entity"])
                for route in POSITIVE:
                    plans[f"{phase}|{task['id']}|{route}"] = request_for(route,v1.ANSWER_SYSTEM,payload_for(task,memory),v1.ANSWER_SCHEMA)
    return plans


class Client:
    def __init__(self,out):
        self.out,self.calls,self.tokens,self.failures = out,0,0,0
        self.start = time.perf_counter()
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def metadata(self):
        def get(path,body=None):
            request = urllib.request.Request(v1.URL+path,data=v1.packed(body).encode() if body else None,
                                             headers={"Content-Type":"application/json"})
            with self.opener.open(request,timeout=10) as response:
                return json.load(response)
        tags = get("/api/tags")
        if next(m for m in tags["models"] if m["name"] == CONFIG["model"])["digest"] != CONFIG["digest"]:
            raise ValueError("Model digest mismatch")
        return dict(tags=tags,version=get("/api/version"),show=get("/api/show",dict(model=CONFIG["model"])),
                    python=platform.python_version())

    def budget(self):
        remaining = CONFIG["max_seconds"]-(time.perf_counter()-self.start)
        if self.calls >= CONFIG["max_calls"] or self.tokens >= CONFIG["max_tokens"] or remaining <= 0 or self.failures >= 3:
            raise RuntimeError("Run budget or infrastructure failure ceiling reached")
        return min(60,remaining)

    def call(self,key,plan,row):
        timeout = self.budget()
        self.calls += 1
        row["calls"] = 1
        start = time.perf_counter()
        log = dict(key=key,call=self.calls,started_utc=timestamp(),**plan)
        try:
            req = urllib.request.Request(v1.URL+plan["endpoint"],data=v1.packed(plan["request"]).encode(),headers={"Content-Type":"application/json"})
            with self.opener.open(req,timeout=timeout) as response:
                result = json.load(response)
            log["response"] = result
            row["prompt_tokens"],row["output_tokens"] = result["prompt_eval_count"],result["eval_count"]
            row["cached_prompt_tokens"] = result.get("prompt_eval_cached_count")
            self.tokens += row["prompt_tokens"]+row["output_tokens"]
            if not result.get("done") or result.get("done_reason") == "length" or row["prompt_tokens"]+row["output_tokens"] > CONFIG["num_ctx"] or time.perf_counter()-start > 60:
                raise RuntimeError("Completion, truncation or episode budget failure")
            raw = result["response"] if plan["endpoint"] == "/api/generate" else result["message"]["content"]
            if row["phase"] == "controls":
                action = json.loads(raw)
                if not isinstance(action,dict) or set(action) != {"answer"} or not isinstance(action["answer"],str):
                    raise ValueError("Invalid nonce action")
            else:
                action = v1.parse(raw)
            return action
        except Exception as error:
            row.update(valid=False,error=f"{type(error).__name__}: {error}")
            log["error"] = row["error"]
            if not isinstance(error,(ValueError,KeyError)):
                self.failures += 1
            return None
        finally:
            log["wall_seconds"] = time.perf_counter()-start
            row["model_seconds"] = log["wall_seconds"]
            v1.append(self.out/"generation_calls.jsonl",log)


def summary_for(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["phase"],row["suite"],row["route"])].append(row)
    result = {}
    for (phase,suite,route),items in groups.items():
        entry = {key:sum(r[key] for r in items) for key in ("correct","valid","stale_use","evidence_valid","calls",
                   "prompt_tokens","output_tokens","model_seconds","wall_seconds","search_seconds","memory_bytes")}
        cached = [r["cached_prompt_tokens"] for r in items if r["cached_prompt_tokens"] is not None]
        entry.update(n=len(items),cached_prompt_tokens=sum(cached) if cached else None,cached_count_available_episodes=len(cached))
        result.setdefault(phase,{}).setdefault(suite,{})[route] = entry
    return result


def candidates_from(summary):
    return [route for route in POSITIVE if summary["controls"]["nonce"][route]["correct"] >= 7
            and all(summary["development"][suite][route]["correct"] >= 9
                    and summary["development"][suite][route]["stale_use"] == 0 for suite in ("access","dispatch"))]


def selection(summary,candidates):
    supported = [route for route in candidates if all(summary["confirmation"][suite][route]["correct"] >= 9
                 and summary["confirmation"][suite][route]["stale_use"] == 0 for suite in ("access","dispatch"))]
    return next((route for route in ("chat_system","raw_system","chat_user") if route in supported),None),supported


def new_row(phase,task,route):
    return dict(phase=phase,task_id=task["id"],suite=task.get("suite","nonce"),scenario=task.get("scenario","nonce"),
                route=route,calls=0,prompt_tokens=0,output_tokens=0,cached_prompt_tokens=None,
                model_seconds=0,search_seconds=0,memory_bytes=0,valid=True,stale_use=False,evidence_valid=False)


def paired(rows):
    output = {}
    for phase in ("development","confirmation"):
        for suite in ("access","dispatch"):
            indexed = {(r["task_id"],r["route"]):r for r in rows if r["phase"] == phase and r["suite"] == suite}
            ids = sorted({key[0] for key in indexed})
            if not ids:
                continue
            for route in ("chat_user","raw_system"):
                differences = [int(indexed[(i,route)]["correct"])-int(indexed[(i,"chat_system")]["correct"]) for i in ids]
                wins,losses = differences.count(1),differences.count(-1)
                n = wins+losses
                p = min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
                output[f"{phase}/{suite}/{route}_vs_chat_system"] = dict(wins=wins,losses=losses,ties=differences.count(0),
                                                                                         descriptive_exact_p=p)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return
    data = make_data()
    plans = plan_requests(data)
    out = ROOT/"results"/datetime.now(timezone.utc).strftime("instruction_delivery_%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True)
    for name in ("INSTRUCTION_DELIVERY_PROTOCOL.md","instruction_delivery.py","test_instruction_delivery.py","experiment.py"):
        shutil.copy2(ROOT/name,out/name)
    v1.save(out/"dataset_evaluator_only.json",data)
    v1.save(out/"planned_requests.json",plans)
    manifest = dict(status="starting",created_utc=timestamp(),config=CONFIG,dataset_sha256=v1.digest(data),
                    frozen_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()},
                    cloud_inference_calls=0,previous_evaluation_used=False)
    v1.save(out/"manifest.json",manifest)
    client,rows,gates = Client(out),[],{}
    print(str(out),flush=True)
    try:
        v1.save(out/"model_metadata.json",client.metadata())
        for task in data["controls"]:
            order = POSITIVE+["chat_none"]
            random.Random(task["id"]).shuffle(order)
            for route in order:
                row = new_row("controls",task,route)
                start = time.perf_counter()
                action = client.call(f"controls|{task['id']}|{route}",plans[f"controls|{task['id']}|{route}"],row)
                row.update(action=action,correct=bool(action and action["answer"] == ("UNKNOWN" if route == "chat_none" else task["marker"])),
                           wall_seconds=time.perf_counter()-start)
                v1.append(out/"episodes.jsonl",row)
                rows.append(row)
        control = summary_for(rows)["controls"]["nonce"]
        gates["controls_pass"] = control["chat_none"]["correct"] >= 7 and any(control[r]["correct"] >= 7 for r in POSITIVE)
        print(v1.packed(control),flush=True)
        if not gates["controls_pass"]:
            manifest["status"] = "stopped_after_controls"
        else:
            for phase,suites in data["phases"].items():
                if phase == "confirmation" and not gates.get("candidates"):
                    break
                for suite,dataset in suites.items():
                    store = v1.PassiveStore(dataset["corpus"])
                    before = v1.digest(store.read_all())
                    for task in dataset["tasks"]:
                        order = POSITIVE+["symbolic"]
                        random.Random(task["id"]).shuffle(order)
                        for route in order:
                            if client.failures >= 3 or time.perf_counter()-client.start >= CONFIG["max_seconds"]:
                                raise RuntimeError("Run time/failure ceiling reached")
                            row = new_row(phase,task,route)
                            start = time.perf_counter()
                            memory = store.search(task["observation"]["entity"])
                            row.update(search_seconds=time.perf_counter()-start,memory_bytes=len(v1.packed(memory).encode()),
                                       query=task["observation"]["entity"],returned_ids=[r["id"] for r in memory])
                            action = (v1.resolve(task["observation"],task["observation"]["current"]+memory) if route == "symbolic"
                                      else client.call(f"{phase}|{task['id']}|{route}",plans[f"{phase}|{task['id']}|{route}"],row))
                            row.update(v1.score(task,action,task["observation"]["current"]+memory))
                            row.update(action=action,wall_seconds=time.perf_counter()-start)
                            v1.append(out/"episodes.jsonl",row)
                            rows.append(row)
                            if route == "symbolic" and not row["correct"]:
                                raise RuntimeError("Symbolic reference failed")
                    if before != v1.digest(store.read_all()):
                        raise RuntimeError("Passive corpus changed")
                    print(f"{phase} {suite} complete; calls={client.calls}",flush=True)
                if phase == "development":
                    gates["candidates"] = candidates_from(summary_for(rows))
                    v1.save(out/"gates.json",gates)
                    print(v1.packed(gates),flush=True)
            if "confirmation" in summary_for(rows):
                gates["selected_route"],gates["supported_routes"] = selection(summary_for(rows),gates["candidates"])
                manifest["status"] = "confirmed_route" if gates["selected_route"] else "confirmation_failed"
            else:
                manifest["status"] = "stopped_after_development"
    except Exception as error:
        manifest.update(status="stopped_on_failure",error=f"{type(error).__name__}: {error}")
    finally:
        manifest.update(calls=client.calls,actual_tokens=client.tokens,infrastructure_failures=client.failures,
                        finished_utc=timestamp(),wall_seconds=time.perf_counter()-client.start)
        v1.save(out/"manifest.json",manifest)
        v1.save(out/"summary.json",summary_for(rows))
        v1.save(out/"gates.json",gates)
        v1.save(out/"paired_comparisons.json",paired(rows))
        print(v1.packed(dict(status=manifest["status"],calls=client.calls,gates=gates)),flush=True)


if __name__ == "__main__":
    main()
