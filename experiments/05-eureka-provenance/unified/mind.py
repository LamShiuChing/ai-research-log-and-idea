"""The unified system: one class wiring every validated organ together.

  core    - the E9-consolidated GPT (skills live in frozen weights)
  memory  - editable, PERSISTENT kNN store of question->answer pairs (E6/E5):
            remember() saves to memory.json; survives shutdown; human-editable
  tools   - the arithmetic oracle audits math answers at inference (E8)
  novelty - self-calibrated distance detector (E8): unknown skills are declined
            honestly and logged to backlog.jsonl for future adapter training

Run from lab/gpt (data paths in checkpoints are relative to it).
"""
import json
import operator
import os
import pickle
import random
import re
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lab", "gpt"))
from model import GPT, GPTConfig

OPS = {"plus": operator.add, "minus": operator.sub, "times": operator.mul}


def oracle(q):
    m = re.search(r"(\d+) (plus|minus|times) (\d+)", q)
    return str(OPS[m.group(2)](int(m.group(1)), int(m.group(3)))) if m else None


class Mind:
    def __init__(self, core_dir, seed_memories, memory_path=os.path.join(HERE, "memory.json"),
                 lam=0.75, k=8):
        self.lam, self.k = lam, k
        self.memory_path = memory_path
        ckpt = torch.load(os.path.join(core_dir, "ckpt.pt"), map_location="cuda", weights_only=True)
        self.core = GPT(GPTConfig(**ckpt["model_args"]))
        self.core.load_state_dict({kk.removeprefix("_orig_mod."): v for kk, v in ckpt["model"].items()})
        self.core.eval().cuda()
        dataset = ckpt["config"]["dataset"]
        meta_dir = dataset if os.path.isdir(dataset) else os.path.join("data", dataset)
        with open(os.path.join(meta_dir, "meta.pkl"), "rb") as f:  # generated locally — trusted
            meta = pickle.load(f)
        self.stoi, self.itos = meta["stoi"], meta["itos"]
        self.memories = dict(seed_memories)
        self._seed = set(self.memories)
        if os.path.exists(memory_path):  # persistence: what was remembered stays remembered
            with open(memory_path, encoding="utf-8") as f:
                self.memories.update(json.load(f))
        self._rebuild_store()
        self._calibrate()

    # --- memory (E6, persistent) ---
    @torch.no_grad()
    def _rebuild_store(self):
        keys, vals, qfeats = [], [], []
        for q, a in self.memories.items():
            ids = [self.stoi[ch] for ch in f"Q: {q}? A: {a}.\n" if ch in self.stoi]
            h = self.core._trunk(torch.tensor([ids], device="cuda"))[0]
            keys.append(h[:-1])
            vals.append(torch.tensor(ids[1:], device="cuda"))
            qfeats.append(self._feats(q))
        self.keys, self.vals = torch.cat(keys), torch.cat(vals)
        self.mem_feats = torch.stack(qfeats)

    def remember(self, question, answer):
        """New knowledge enters instantly, survives shutdown, trains nothing."""
        self.memories[question.strip().rstrip("?")] = answer.strip().rstrip(".")
        self._rebuild_store()
        learned = {q: a for q, a in self.memories.items() if q not in self._seed}
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump(learned, f, indent=1, ensure_ascii=False)

    # --- self-awareness (E8) ---
    @torch.no_grad()
    def _feats(self, q):
        ids = [self.stoi[c] for c in f"Q: {q}? A:" if c in self.stoi]
        return self.core._trunk(torch.tensor([ids], device="cuda"))[0, -1]

    def _calibrate(self):
        random.seed(71)
        arith = [f"What is {random.randint(10, 99)} plus {random.randint(10, 99)}" for _ in range(20)] + \
                [f"What is {random.randint(20, 99)} minus {random.randint(10, 19)}" for _ in range(20)] + \
                [f"What is {random.randint(2, 99)} times {random.randint(2, 9)}" for _ in range(20)]
        self.skill_feats = torch.cat([self.mem_feats, torch.stack([self._feats(q) for q in arith])])
        fresh = random.sample(list(self.memories), min(20, len(self.memories))) + \
                [f"What is {random.randint(10, 99)} plus {random.randint(10, 99)}" for _ in range(10)]
        self.tau = max(self._dist(q, self.skill_feats) for q in fresh) * 1.1

    def _dist(self, q, ref):
        return torch.cdist(self._feats(q)[None], ref).min().item()

    # --- generation: core + memory blend ---
    @torch.no_grad()
    def _generate(self, q, memory):
        ids = [self.stoi[c] for c in f"Q: {q}? A:" if c in self.stoi]
        n = len(ids)
        for _ in range(16):
            h = self.core._trunk(torch.tensor([ids], device="cuda"))[0, -1]
            p = F.softmax(self.core.lm_head(h), dim=-1)
            if memory:
                d = torch.cdist(h[None], self.keys)[0]
                nn = torch.topk(-d, min(self.k, len(self.vals))).indices
                w = F.softmax(-d[nn], dim=-1)
                p = self.lam * torch.zeros_like(p).index_add_(0, self.vals[nn], w) + (1 - self.lam) * p
            nxt = int(p.argmax())
            ids.append(nxt)
            if self.itos[nxt] == ".":
                break
        return "".join(self.itos[i] for i in ids[n:]).split(".")[0].strip()

    # --- the one entry point ---
    def answer(self, q):
        q = q.strip().rstrip("?")
        truth = oracle(q)
        if truth is not None:  # tool-verifiable: core answers, the tool audits
            a = self._generate(q, memory=False)
            if a == truth:
                return {"answer": a, "organ": "core (tool-verified)"}
            return {"answer": truth, "organ": f"tool-corrected (core said {a!r})"}
        if self._dist(q, self.mem_feats) <= self.tau:  # something remembered is nearby
            return {"answer": self._generate(q, memory=True), "organ": "core + memory"}
        if self._dist(q, self.skill_feats) > self.tau:  # not one of my skills: decline, log
            with open(os.path.join(HERE, "backlog.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"q": q}) + "\n")
            return {"answer": "I don't have this skill yet — logged it to learn.",
                    "organ": "novelty → backlog"}
        return {"answer": self._generate(q, memory=True), "organ": "core + memory"}
