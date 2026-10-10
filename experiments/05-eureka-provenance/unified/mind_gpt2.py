"""The Mind on an adopted core: GPT-2 124M (fully open weights) inside our fork.

  core   - GPT-2 loaded via from_pretrained into lab/gpt/model.py (all organs available)
  memory - persistent kNN store (E6): remember() -> memory_gpt2.json, survives shutdown
  tools  - the arithmetic oracle audits math answers at inference (E8)
  chat   - everything else: sampled generation. GPT-2-class quality: fluent, often silly.

Run from lab/gpt.
"""
import json
import operator
import os
import re
import sys
from collections import deque

import tiktoken
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lab", "gpt"))
from model import GPT, GPTConfig

OPS = {"plus": operator.add, "minus": operator.sub, "times": operator.mul}
NL = 198  # GPT-2 newline token


def oracle(q):
    m = re.search(r"(\d+) (plus|minus|times) (\d+)", q)
    return str(OPS[m.group(2)](int(m.group(1)), int(m.group(3)))) if m else None


STOP = {"what", "is", "my", "do", "i", "the", "a", "an", "of", "who", "which", "where", "our", "your"}


def content(q):
    return set(q.lower().replace("what's", "what is").replace("?", "").split()) - STOP


class MindGPT2:
    def __init__(self, model_type="gpt2", memory_path=os.path.join(HERE, "memory_gpt2.json"),
                 lam=0.8, k=8):
        self.lam, self.k = lam, k
        self.memory_path = memory_path
        self.history = deque(maxlen=6)  # short-term conversation window
        self.core = GPT.from_pretrained(model_type)
        self.core.to(torch.bfloat16).eval().cuda()
        self.chat_core, self.chat_tuned = self.core, False
        chat_ckpt = os.path.join({"gpt2": "out-e11-chat", "gpt2-xl": "out-e11-chat-xl"}[model_type],
                                 "ckpt.pt")
        if os.path.exists(chat_ckpt):  # E11: conversation adapter on the frozen core
            ck = torch.load(chat_ckpt, map_location="cuda", weights_only=True)
            m = GPT(GPTConfig(**ck["model_args"]))
            m.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ck["model"].items()})
            m.to(torch.bfloat16).eval().cuda()
            self.chat_core, self.chat_tuned = m, True
        self.enc = tiktoken.get_encoding("gpt2")
        self.memories = {"What is your name": "Mind"}
        self._seed = set(self.memories)
        if os.path.exists(memory_path):
            with open(memory_path, encoding="utf-8") as f:
                self.memories.update(json.load(f))
        self._rebuild_store()

    # --- memory (E6, persistent) ---
    @torch.no_grad()
    def _rebuild_store(self):
        keys, vals = [], []
        for q, a in self.memories.items():
            ids = self.enc.encode(f"Q: {q}? A: {a}.\n")
            h = self.core._trunk(torch.tensor([ids], device="cuda"))[0]
            keys.append(h[:-1])
            vals.append(torch.tensor(ids[1:], device="cuda"))
        self.keys, self.vals = torch.cat(keys), torch.cat(vals)

    def remember(self, question, answer):
        self.memories[question.strip().rstrip("?")] = answer.strip().rstrip(".")
        self._rebuild_store()
        learned = {q: a for q, a in self.memories.items() if q not in self._seed}
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(learned, f, indent=1, ensure_ascii=False)

    def _recall(self, q):
        """A stored question matches only if it covers every content word of the query."""
        want = content(q)
        hits = [m for m in self.memories if want and want <= content(m)]
        return min(hits, key=lambda m: len(content(m))) if hits else None

    # --- generation ---
    @torch.no_grad()
    def _memory_answer(self, question):
        ids = self.enc.encode(f"Q: {question}? A:")
        n = len(ids)
        for _ in range(24):
            h = self.core._trunk(torch.tensor([ids], device="cuda"))[0, -1]
            p = F.softmax(self.core.lm_head(h), dim=-1)
            d = torch.cdist(h[None], self.keys)[0]
            nn = torch.topk(-d, min(self.k, len(self.vals))).indices
            w = F.softmax(-d[nn], dim=-1)
            p = self.lam * torch.zeros_like(p).index_add_(0, self.vals[nn], w) + (1 - self.lam) * p
            nxt = int(p.argmax())
            if nxt == NL:
                break
            ids.append(nxt)
            if "." in self.enc.decode([nxt]):
                break
        return self.enc.decode(ids[n:]).split(".")[0].strip()

    @torch.no_grad()
    def _greedy_qa(self, question):
        ids = self.enc.encode(f"Q: {question}? A:")
        out = self.core.generate(torch.tensor([ids], device="cuda"), max_new_tokens=12, top_k=1)
        return self.enc.decode(out[0].tolist()[len(ids):]).split("\n")[0].split(".")[0].strip()

    @torch.no_grad()
    def _chat(self, q):
        convo = "".join(f"User: {u}\nMind: {m}\n" for u, m in self.history)
        if self.chat_tuned:
            prompt = f"{convo}User: {q}\nMind:"  # matches the adapter's training format
        else:
            prompt = f"The following is a conversation with Mind, a helpful assistant.\n{convo}User: {q}\nMind:"
        ids = self.enc.encode(prompt)
        n = len(ids)
        out = self.chat_core.generate(torch.tensor([ids], device="cuda"),
                                      max_new_tokens=80, temperature=0.8, top_k=50)
        text = self.enc.decode(out[0].tolist()[n:])
        return text.split("\nUser")[0].split("\n\n")[0].strip() or text.strip()

    def note(self, q, a):
        """Feed a turn into the short-term conversation window."""
        self.history.append((q, a))

    def buffer(self, grade, q, a):
        """Verified exchanges accumulate as feedstock for future micro-LoRA training."""
        with open(os.path.join(HERE, "buffer.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"grade": grade, "q": q, "a": a}) + "\n")

    # --- the one entry point ---
    def answer(self, q):
        q = q.strip().rstrip("?")
        truth = oracle(q)
        if truth is not None:
            a = self._greedy_qa(q)
            r = ({"answer": a, "organ": "core (tool-verified)"} if a == truth
                 else {"answer": truth, "organ": f"tool-corrected (core said {a!r})"})
            self.buffer("G0", q, truth)
        else:
            hit = self._recall(q)
            if hit:
                r = {"answer": self._memory_answer(hit), "organ": "core + memory"}
            elif re.match(r"(what|who|which|where)\b", q.lower()) and re.search(r"\bmy\b", q.lower()):
                # a personal question with nothing in memory: never invent facts about the user
                r = {"answer": "I don't have that in my memory yet — tell me and I'll remember.",
                     "organ": "memory (not found — honest decline)"}
            else:
                organ = "core + chat adapter" if self.chat_tuned else "core (GPT-2 chat)"
                r = {"answer": self._chat(q), "organ": organ}
        self.note(q, r["answer"])
        return r
