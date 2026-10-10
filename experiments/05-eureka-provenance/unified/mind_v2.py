"""The Mind v2: eureka organs on a modern, fully-open instruct core (SmolLM3-3B).

  core    - HuggingFaceTB/SmolLM3-3B (Apache 2.0, open weights + open data recipe)
  memory  - the SAME persistent store as v1 (memory_gpt2.json): facts survive a
            core transplant. Claims carry source + verification grade (store.py);
            the relevant slice is injected into the system prompt with its grade label.
  tools   - the arithmetic oracle audits math answers at inference
  honesty - the system prompt forbids inventing personal facts not in memory

Run from anywhere; models load via HF transformers.
"""
import json
import operator
import os
import re
from collections import deque

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from store import Store

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"

OPS = {"plus": operator.add, "minus": operator.sub, "times": operator.mul}


def oracle(q):
    q = re.sub(r"(\d)\s*[*x×]\s*(\d)", r"\1 times \2", q)
    q = re.sub(r"(\d)\s*\+\s*(\d)", r"\1 plus \2", q)
    q = re.sub(r"(\d)\s*-\s*(\d)", r"\1 minus \2", q)
    m = re.search(r"(\d+) (plus|minus|times) (\d+)", q)
    return str(OPS[m.group(2)](int(m.group(1)), int(m.group(3)))) if m else None


class MindV2:
    def __init__(self, memory_path=os.path.join(HERE, "memory_gpt2.json")):
        self.tok = AutoTokenizer.from_pretrained(MODEL_ID)
        self.core = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        self.core.eval().cuda()
        self.history = deque(maxlen=8)
        self._embed_cache = {}
        self.store = Store(memory_path)

    def forget(self, topic):
        hits = self.store.forget(topic)
        for m in hits:
            self._embed_cache.pop(m, None)
        return hits

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

    def note(self, q, a):
        self.history.append({"role": "user", "content": q})
        self.history.append({"role": "assistant", "content": a})

    def buffer(self, grade, q, a, core_ok=None):
        """core_ok=False entries are genuine failures — the sleep daemon's training trigger."""
        rec = {"grade": grade, "q": q, "a": a}
        if core_ok is not None:
            rec["core_ok"] = core_ok
        with open(os.path.join(HERE, "buffer.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _system(self, q):
        rel = self._relevant(q)
        self.last_facts = [m for m, *_ in rel]  # which memories this answer consulted
        facts = "\n".join(f"- {m}? {a} [{label}]" for m, a, label in rel) or "- (none yet)"
        guess = ("A fact tagged 'unconfirmed' is only a guess: say \"I think …, is that right?\". "
                 if any("unconfirmed" in label for *_, label in rel) else "")
        return ("/no_think\nYou are Mind, a concise personal assistant running locally. "
                "Known facts about your user, each tagged with how it was verified:\n" + facts +
                "\nFacts tagged 'verified' or 'you told me' are certain: state them directly, "
                "never add caveats or ask the user to verify them. For a fact from a source, "
                "name the source. " + guess +
                "If asked a personal fact that is not in this list, say you don't know it yet "
                "and ask to be told. Never invent facts about the user. Keep replies short and natural.")

    @torch.no_grad()
    def _chat(self, q):
        msgs = [{"role": "system", "content": self._system(q)}] + list(self.history) + \
               [{"role": "user", "content": q}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to("cuda")
        out = self.core.generate(**enc, max_new_tokens=200, do_sample=True,
                                 temperature=0.6, top_p=0.9, repetition_penalty=1.1,
                                 pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

    # --- the thinking organ: reasoning as verified chains ---
    OP_TOKENS = re.compile(r"(?<=\d)\s*[*+x×-]\s*(?=\d)|\b(?:plus|minus|times)\b", re.I)
    PROBLEM_WORDS = re.compile(r"\bhow many|total|left|each|altogether|then\b", re.I)

    def needs_reasoning(self, q):
        nums = len(re.findall(r"\d+", q))
        return nums >= 2 and (len(self.OP_TOKENS.findall(q)) >= 2 or
                              bool(self.PROBLEM_WORDS.search(q)))

    @torch.no_grad()
    def _step(self, q, steps):
        done = "\n".join(f"{e} = {v} (verified)" for e, v in steps) or "(none yet)"
        msgs = [{"role": "system", "content":
                 "/no_think\nYou are the planner of a calculator system. You CANNOT do arithmetic "
                 "yourself — every calculation must go through the calculator, one at a time.\n"
                 "If any calculation is still needed, reply EXACTLY: CALC: <number> <plus|minus|times> <number>\n"
                 "Only when the verified list already contains every needed result, reply: "
                 "DONE: <final answer using a verified number>"},
                {"role": "user", "content": f"Problem: {q}\nVerified so far:\n{done}"}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to("cuda")
        out = self.core.generate(**enc, max_new_tokens=40, do_sample=False,
                                 pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def reason(self, q):
        """The core plans one step at a time; the oracle executes every step; the
        chain is auditable and the final answer must carry a verified value."""
        steps = []
        for _ in range(5):
            out = self._step(q, steps)
            m = re.search(r"CALC:\s*(\d+\s*(?:plus|minus|times|[*+x×-])\s*\d+)", out, re.I)
            if m:
                expr = re.sub(r"\s+", " ", m.group(1)).strip()
                val = oracle(expr)
                if val is None or (steps and (expr, val) == steps[-1]):
                    break
                steps.append((expr, val))
                self.buffer("G0", expr, val)
                continue
            d = re.search(r"DONE:\s*(.+)", out, re.I | re.S)
            if d and steps:
                ans = d.group(1).strip().split("\n")[0]
                if steps[-1][1] not in ans:  # the answer must carry the verified value
                    ans = steps[-1][1]
                chain = " → ".join(f"{e}={v}" for e, v in steps)
                return {"answer": ans, "organ": f"reasoning (verified chain: {chain})"}
            break
        return None

    @torch.no_grad()
    def extract_fact(self, q):
        """Ask the core itself whether the user just stated a personal fact (typo-tolerant,
        context-aware). Returns (question, answer) or None."""
        prev = self.history[-1]["content"] if self.history else ""
        msgs = [{"role": "system",
                 "content": "/no_think\nYou extract personal facts from chat. Reply ONLY with JSON, nothing else."},
                {"role": "user",
                 "content": f'Assistant said: "{prev}"\nUser replied: "{q}"\n'
                            "Only if the user explicitly states a fact about themselves or their life "
                            "(name, favorites, family, hometown, job), reply "
                            '{"question": "What is my <thing>", "answer": "<value copied from the user\'s words>"}. '
                            "Requests, questions, opinions about you, and topics of discussion are NOT facts. "
                            "If unsure, reply null."}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True).to("cuda")
        out = self.core.generate(**enc, max_new_tokens=60, do_sample=False,
                                 pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        m = re.search(r'\{[^{}]*"question"[^{}]*\}', text)
        if m:
            try:
                d = json.loads(m.group(0))
                a = str(d.get("answer") or "")
                if d.get("question") and a and a.lower() in q.lower() and len(a.split()) <= 3:  # the user's own words, not the whole message
                    return str(d["question"]).replace("your", "my").rstrip("?"), a
            except json.JSONDecodeError:
                pass
        return None

    def answer(self, q):
        self.last_facts = []
        needs = self.needs_reasoning(q)
        r = self.reason(q) if needs else None
        if r is None and needs:  # multi-step problem the chain couldn't verify: never fake an audit
            r = {"answer": self._chat(q), "organ": "core (unverified — reasoning fallback)"}
        if r is None:
            truth = oracle(q.strip().rstrip("?"))
            if truth is not None:
                a = self._chat(q)
                digits = re.findall(r"-?\d[\d,]*", a.replace(",", ""))
                ok = truth in digits
                r = ({"answer": a, "organ": "core (tool-verified)"} if ok
                     else {"answer": truth, "organ": f"tool-corrected (core said {a!r})"})
                self.buffer("G0", q, truth, core_ok=ok)
            else:
                r = {"answer": self._chat(q), "organ": "core + memory prompt"}
        self.note(q, r["answer"])
        return r
