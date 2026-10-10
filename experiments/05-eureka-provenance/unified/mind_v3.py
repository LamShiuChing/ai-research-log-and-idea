"""The Mind v3: the core drives everything through its native tool calls.

  core   - HuggingFaceTB/SmolLM3-3B decides for itself when to calculate, remember, forget
  memory - store.py: graded, provenance-tagged; the relevant slice is shown in the prompt
  truth  - calc() is the only arithmetic. An answer is 'verified' only when every number in
           it came out of the calculator or the user's own question. Grades come from
           evidence (is the value in the user's words?), never from the model.

No rules about English anywhere: no routing regexes, no extraction templates, no command syntax.
"""
import ast
import json
import operator
import os
import re
from collections import deque

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from store import GRADE, LABEL, Store

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"

TOOLS = [json.dumps({"type": "function", "function": f}) for f in [
    {"name": "recall",
     "description": "Look up what is known about the user. Call it before answering any question about "
                    "the user (their name, preferences, family, places, work).",
     "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "what to look up"}},
                    "required": ["query"]}},
    {"name": "calc",
     "description": "Evaluate an arithmetic expression exactly. Use it for every calculation, however "
                    "small; never do arithmetic yourself.",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "e.g. 23*17-95"}},
                    "required": ["expression"]}},
    {"name": "remember",
     "description": "Save a fact the user just stated about themselves or their life (name, family, "
                    "favorites, places). Not for requests, questions, or topics of discussion.",
     "parameters": {"type": "object", "properties": {
         "question": {"type": "string", "description": "the question this fact answers, e.g. 'What is my name'"},
         "answer": {"type": "string", "description": "the value, in the user's own words"}},
         "required": ["question", "answer"]}},
    {"name": "forget",
     "description": "Delete memories about a topic when the user asks you to forget it.",
     "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}},
]]

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
       ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg}
NUM = re.compile(r"\d+(?:\.\d+)?")


def calc(expression):
    """Exact arithmetic over literals and + - * / // % ** — nothing else can run."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp):
            return OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp):
            return OPS[type(n.op)](ev(n.operand))
        raise ValueError(expression)
    tree = ast.parse(expression.replace("×", "*").replace("÷", "/").replace("^", "**"), mode="eval").body
    if not isinstance(tree, ast.BinOp):
        raise ValueError(expression)  # a bare number is not a calculation — nothing to verify
    v = ev(tree)
    return str(int(v)) if float(v).is_integer() else str(v)


CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>|(\{\s*\"name\"\s*:.*?\"arguments\"\s*:\s*\{.*?\}\s*\})", re.S)


def tool_call(text):
    """(name, arguments) from the model's <tool_call> block — or a bare call object, which
    the 3B core sometimes emits without the tags — else None."""
    m = CALL.search(text)
    try:
        d = json.loads(m.group(1) or m.group(2)) if m else None
    except json.JSONDecodeError:
        return None
    return (d["name"], d.get("arguments") or {}) if d and "name" in d else None


def numbers(text):
    return set(NUM.findall(text.replace(",", "")))


class MindV3:
    def __init__(self, memory_path=os.path.join(HERE, "memory_gpt2.json"), adapter=None):
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.core = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
        if adapter:  # a LoRA trained on verified tool traces (train_v3.py)
            from peft import PeftModel
            self.core = PeftModel.from_pretrained(self.core, adapter)
        self.core.eval().cuda()
        self.history = deque(maxlen=8)
        self._embed_cache = {}
        self.store = Store(memory_path)

    # --- retrieval memory: only the relevant slice enters the prompt ---
    @torch.no_grad()
    def _embed(self, text):
        enc = self.tok(text, return_tensors="pt").to("cuda")
        h = self.core(**enc, output_hidden_states=True).hidden_states[-1]
        v = h.mean(1)[0].float()
        return v / v.norm()

    def _relevant(self, q, k=6):
        qs = list(self.store)
        if len(qs) > k:
            for m in qs:
                if m not in self._embed_cache:
                    self._embed_cache[m] = self._embed(m)
            qv = self._embed(q)
            qs = sorted(qs, key=lambda m: -(qv @ self._embed_cache[m]).item())[:k]
        return self.store.facts(qs)

    SYSTEM = ("/no_think\nYou are Mind, a concise personal assistant running locally. Keep replies short. "
              "Answer questions about the user only from the known facts below (or what recall returns); "
              "if none applies, say you don't know that yet and ask. Facts tagged 'you told me' or "
              "'verified' are certain: state them directly. A fact tagged 'unconfirmed' is a guess to "
              "confirm. Never invent facts about the user; general-knowledge questions are fine to answer. "
              "Use calc for any arithmetic; remember when the user tells you something about themselves; "
              "forget when asked.")

    def _system(self, q):
        rel = self._relevant(q)
        self.last_facts = [m for m, *_ in rel]
        facts = "\n".join(f"- {m}? {a} [{label}]" for m, a, label in rel) or "- (none yet)"
        return self.SYSTEM + "\nKnown facts about your user, each tagged with how it was verified:\n" + facts

    @torch.no_grad()
    def _generate(self, msgs):
        enc = self.tok.apply_chat_template(msgs, xml_tools=TOOLS, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to("cuda")
        torch.manual_seed(0)  # model-card sampling (0.6 / 0.95), reproducible per prompt
        out = self.core.generate(**enc, max_new_tokens=200, do_sample=True, temperature=0.6, top_p=0.95,
                                 pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=False)
        return text.split("<|im_end|>")[0].strip()

    def _run(self, name, args, q):
        if name == "recall":
            rel = self._relevant(str(args.get("query", q)))
            self.last_facts += [m for m, *_ in rel]
            return "\n".join(f"{m}? {a} [{label}]" for m, a, label in rel) or "nothing known about that"
        if name == "calc":
            try:
                return calc(str(args.get("expression", "")))
            except Exception:
                return "error: not an arithmetic expression"
        if name == "remember":
            question = str(args.get("question", "")).replace("your", "my").replace("the user's", "my")
            answer = str(args.get("answer", ""))
            if not (question and answer):
                return "error: question and answer required"
            if answer.lower() not in q.lower():  # evidence gate: the store holds only what the user said
                return "not saved: the user did not say that"
            self.store.remember(question, answer)
            return f"saved: {question} = {answer} [{LABEL[GRADE['user']]}]"
        if name == "forget":
            gone = self.store.forget(str(args.get("topic", "")))
            for m in gone:
                self._embed_cache.pop(m, None)
            return f"forgot: {'; '.join(gone) or 'nothing matched'}"
        return f"error: unknown tool {name}"

    def _organ(self, answer, q, used):
        calls = [f"{n}({', '.join(map(str, a.values()))}) → {r}" for n, a, r in used]
        nums = numbers(answer)
        if nums:
            known = numbers(q).union(*(numbers(r) for n, _, r in used if n == "calc"))
            calls.insert(0, "verified" if nums <= known and any(n == "calc" for n, *_ in used)
                         else "unverified numbers")
        return " · ".join(["core"] + calls)

    def _loop(self, msgs, q):
        """Generate until the core stops calling tools. Returns (answer, calls used)."""
        used, done = [], {}
        for _ in range(4):
            text = self._generate(msgs)
            call = tool_call(text)
            if not call:
                break
            key = json.dumps(call, sort_keys=True)
            result = done.get(key) or self._run(*call, q)  # a repeated call is answered, not re-run
            done[key] = result
            used.append((*call, result))
            msgs += [{"role": "assistant", "content": text},
                     {"role": "tool", "content": f"{call[0]} result: {result}"}]
        answer = re.sub(r"<tool_call>.*?</tool_call>|<think>.*?</think>", "", text, flags=re.S).strip()
        return answer or (used[-1][2] if used else ""), used

    def answer(self, q):
        msgs = [{"role": "system", "content": self._system(q)}] + list(self.history) + [{"role": "user", "content": q}]
        answer, used = self._loop(msgs, q)
        calc_used = any(n == "calc" for n, *_ in used)
        if len(numbers(q)) >= 2 and numbers(answer) - numbers(q) and not calc_used:  # quantities, no check: one re-ask
            retry = msgs + [{"role": "assistant", "content": answer},
                            {"role": "user", "content": "Verify every number with calc, then answer again."}]
            answer2, used2 = self._loop(retry, q)
            if any(n == "calc" for n, *_ in used2):
                answer, used = answer2, used + used2
        self.history += [{"role": "user", "content": q}, {"role": "assistant", "content": answer}]
        return {"answer": answer, "organ": self._organ(answer, q, used), "facts": self.last_facts}
