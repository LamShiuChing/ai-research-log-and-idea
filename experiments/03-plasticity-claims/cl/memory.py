"""A small language memory (the user's idea, 2026-09-03 22:22): the v3 child as the half of a
two-model mind that learns online. A frozen thinker decides what to keep and what to ask; this
model keeps it in its weights and answers in words. remember(sentence) is a nap over the sentence
plus fill-the-blank units whose answers are the sentence's own words (the text-quiz grader,
design-interface §7.7 - the sentence is the answer key); recall(question) answers from the weights
in the Q:/A: register, or says I don't know. The nap is sparse_run.py's direct nap: tag = top-k
|dL/dW| of the rows' own gradient, sparse step on the rows + rehearsal from the child's own diet and
from its earlier memories, exact fold. Bounded: the model never grows. One life dir: weights,
reservoir, memories, journal."""
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from cl import bpe, stages
from cl.gen import sample
from cl.lm import GPT, TokenReservoir
from cl.sparse import SparseNursery

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["aochildes", "switchboard", "bnc_spoken", "open_subtitles", "children_stories", "cbt",
         "qed", "simple_wikipedia", "gutenberg", "wikipedia"]
STAGES = ["qa", "cot", "tools", "idk", "quiz", "wcot"]
STOP = set("a an the is are was were my your his her its our their of to in on at and or for with be it this that".split())


class Memory:
    def __init__(self, life="fun/memory", init="models/child-base-v3.pt", vocab="models/bpe8k.json", track="100M",
                 block=256, batch=16, sleep_old=16, slots=8000, frac=0.02, lr=3e-4, study=40, know=0.5,
                 dev="cuda", seed=0):
        self.life, self.block, self.batch, self.sleep_old = ROOT / life, block, batch, sleep_old
        self.frac, self.lr, self.study, self.know, self.dev = frac, lr, study, know, dev
        self.life.mkdir(parents=True, exist_ok=True)
        self.gen = torch.Generator().manual_seed(seed)
        self.tok = bpe.load(str(ROOT / vocab))
        self.V = self.tok.get_vocab_size()
        self.net = GPT(vocab=self.V, dim=1024, layers=12, heads=16, block=block).to(dev)
        w = self.life / "weights.pt"
        self.net.load_state_dict(torch.load(w if w.exists() else ROOT / init, weights_only=True))
        self.res = TokenReservoir(slots, block, dev)
        r = self.life / "reservoir.pt"
        if r.exists():
            d = torch.load(r, weights_only=True)
            self.res.x, self.res.seen = d["x"].to(dev), d["seen"]
        else:
            self._birth(track, slots)
        m = self.life / "memories.txt"
        self.memories = m.read_text(encoding="utf-8").splitlines() if m.exists() else []

    def _birth(self, track, slots):
        """The child's own diet as its rehearsal: childhood chunks and the format streams (sparse_run.py)."""
        for s in ORDER:
            text = (ROOT / "data/babylm" / track / f"{s}.txt").read_text(encoding="utf-8", errors="ignore")
            for _ in range(slots // 2 // len(ORDER)):
                i = int(torch.randint(len(text) - 10_000, (), generator=self.gen))
                chunk = self.enc(text[i:i + 8000][text[i:i + 8000].find("\n") + 1:][:4000])
                if len(chunk) >= self.block:
                    self.res.add(chunk[:self.block].to(self.dev), self.gen)
            del text
        for st in STAGES:
            n = slots // len(STAGES) // 2 * self.block
            pb = self.enc(stages.stream(st, 5 * n, 0, track).decode())[:n]
            for lo in range(0, len(pb) - self.block + 1, self.block):
                self.res.add(pb[lo:lo + self.block].to(self.dev), self.gen)

    def enc(self, s):
        return torch.tensor(self.tok.encode(s).ids, dtype=torch.long)

    def fwd(self, x):
        with torch.autocast("cuda", torch.bfloat16):
            return self.net(x)[0]

    def row_bits(self, logits, x):
        return F.cross_entropy(logits[:, :-1].reshape(-1, self.V).float(), x[:, 1:].reshape(-1),
                               reduction="none").view(len(x), -1).mean(1) / math.log(2)

    @staticmethod
    def units(sentence):
        """The sentence, and one fill-the-blank quiz per content word, each its own row WITHOUT the
        sentence in view - so the only way to answer is from the weights. The sentence is the key."""
        s = sentence.strip().rstrip(".")
        w = s.split()
        quizzes = [f"Q: fill the blank: {' '.join('___' if j == i else u for j, u in enumerate(w))}\nA: {t}\n"
                   for i, t in enumerate(w) if t.lower() not in STOP and len(t) > 2]
        return [f"{s}\n"] + quizzes[:5]

    def rows(self, sentence):
        return [self.enc(u)[None].to(self.dev) for u in self.units(sentence)]

    @torch.no_grad()
    def known(self, rows):
        """How surprising the quizzes are with nothing in view: mean bits/token over the rows."""
        return sum(self.row_bits(self.fwd(r), r).item() for r in rows) / len(rows)

    def remember(self, sentence):
        """Study the sentence until it is known (quiz bits <= know) or the study cap: a direct nap
        over its rows, rehearsing the diet and earlier memories every step."""
        t0 = time.time()
        new = self.rows(sentence)
        net, ln2 = self.net, math.log(2)
        net.requires_grad_(True)
        net.zero_grad(set_to_none=True)
        for r in new:
            (F.cross_entropy(self.fwd(r)[:, :-1].reshape(-1, self.V).float(), r[:, 1:].reshape(-1)) / len(new)).backward()
        net.requires_grad_(False)
        before = self.known(new)
        sp = SparseNursery.from_grads(net, self.frac)
        net.zero_grad(set_to_none=True)
        opt = torch.optim.AdamW(sp.params(), lr=self.lr)
        steps = 0
        while steps < self.study and self.known(new) > self.know:
            olds = [r for m in (self.memories[int(torch.randint(len(self.memories), (), generator=self.gen))]
                                for _ in range(min(3, len(self.memories)))) for r in self.rows(m)]
            rows = [self.res.sample(self.sleep_old, self.gen), *new * 3, *olds]
            total = sum(len(r) for r in rows)
            opt.zero_grad()
            for r in rows:
                (self.row_bits(self.fwd(r), r).sum() * ln2 / total).backward()
            opt.step()
            steps += 1
        sp.fold()
        after = self.known(new)
        self.memories.append(sentence.strip())
        (self.life / "memories.txt").write_text("\n".join(self.memories) + "\n", encoding="utf-8")
        entry = dict(t=time.time(), sentence=sentence.strip(), bits=[round(before, 2), round(after, 2)],
                     steps=steps, secs=round(time.time() - t0, 1))
        (self.life / "journal.jsonl").open("a", encoding="utf-8").write(json.dumps(entry) + "\n")
        return entry

    @torch.no_grad()
    def recall(self, question):
        """Answer from the weights alone, in the register; the idk habit says when it has nothing."""
        rows = [self.enc(f"Q: {question.strip().rstrip('?')}\nA:").tolist()]
        n = len(rows[0])
        sample(self.net, rows, 16, None, temp=0)
        return self.tok.decode(rows[0][n:]).split("\n")[0].strip()

    def save(self):
        torch.save(self.net.state_dict(), self.life / "weights.pt")
        torch.save(dict(x=self.res.x.cpu(), seen=self.res.seen), self.life / "reservoir.pt")
