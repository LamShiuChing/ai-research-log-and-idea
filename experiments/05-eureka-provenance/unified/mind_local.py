"""The Mind on a local Ollama model: the same loop, store, tools and tests as v3 — the core
is any Ollama model with tool calling (e.g. qwen2.5:14b). Retrieval embeddings come from
Ollama too (nomic-embed-text). Nothing leaves the machine.
"""
import json
import os
import urllib.request
from collections import deque

import torch

from mind_v3 import HERE, TOOLS, MindV3
from store import Store

OLLAMA = "http://127.0.0.1:11434"  # default local endpoint, github.com/ollama/ollama/blob/main/docs/api.md


class MindLocal(MindV3):
    SYSTEM = MindV3.SYSTEM.removeprefix("/no_think\n")

    def __init__(self, model="qwen2.5:14b", memory_path=os.path.join(HERE, "memory_gpt2.json"),
                 embed_model="nomic-embed-text"):
        self.model, self.embed_model = model, embed_model
        self.tools = [json.loads(t) for t in TOOLS]
        self.history = deque(maxlen=8)
        self._embed_cache = {}
        self.store = Store(memory_path)

    def _post(self, path, obj):
        req = urllib.request.Request(OLLAMA + path, json.dumps(obj).encode(), {"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=600))

    def _embed(self, text):
        v = torch.tensor(self._post("/api/embed", {"model": self.embed_model, "input": text})["embeddings"][0])
        return v / v.norm()

    def _loop(self, msgs, q):
        used, done = [], {}
        for _ in range(4):
            m = self._post("/api/chat", {"model": self.model, "messages": msgs, "tools": self.tools,
                                         "stream": False, "options": {"temperature": 0}})["message"]
            calls = m.get("tool_calls") or []
            if not calls:
                break
            msgs.append(m)
            for c in calls:
                name, args = c["function"]["name"], c["function"].get("arguments") or {}
                key = json.dumps([name, args], sort_keys=True)
                result = done.get(key) or self._run(name, args, q)  # a repeated call is answered, not re-run
                done[key] = result
                used.append((name, args, result))
                msgs.append({"role": "tool", "content": f"{name} result: {result}", "tool_name": name})
        answer = (m.get("content") or "").strip()
        return answer or (used[-1][2] if used else ""), used
