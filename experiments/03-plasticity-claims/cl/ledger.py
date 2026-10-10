"""The audit ledger (notes/design-audit.md, C17): one structure = audit pool + sleep-rehearsal
source. Item = probe slice: n_p stream-confirmed (x, y*) pairs from one domain, with s_ref (base's
score at last audit turn) and a rehearse flag (a canary fraction is audited but never rehearsed).
The stream admits (at merge), the decay law rewrites (rewrite-before-merge on a confirmed
correction), the base never writes. Audit reads m slices per sleep by oldest-audited rotation;
fail = s_now < s_ref - eps. Refresh of s_ref is two-phase so a rolled-back sleep leaves no trace."""
import torch


class Ledger:
    def __init__(self, eps, canary_frac, gen):
        self.eps, self.canary_frac, self.gen = eps, canary_frac, gen
        self.slices = []  # dicts: x, y, domain, s_ref, rehearse, t_audit

    def admit(self, x, y, domain, score, rehearse=None):
        if rehearse is None:
            rehearse = torch.rand((), generator=self.gen).item() >= self.canary_frac
        self.slices.append(dict(x=x, y=y, domain=domain, s_ref=score(x, y), rehearse=rehearse, t_audit=-1))

    def rewrite(self, domain, y_map, score):
        """Rewrite-before-merge: a confirmed correction updates y* and resets s_ref BEFORE the
        merge that carries it, so the audit measures the merge against the corrected expectation."""
        for s in (s for s in self.slices if s["domain"] == domain):
            s["y"] = y_map(s["y"])
            s["s_ref"] = score(s["x"], s["y"])

    def audit(self, m, t, score, extra=()):
        """Score the m oldest-audited slices (plus `extra` slice indices under repair watch).
        Returns (picked indices, scores, failed indices, mean s_ref - s_now over the rotation m)."""
        order = sorted(range(len(self.slices)), key=lambda i: self.slices[i]["t_audit"])
        picked = list(dict.fromkeys([*order[:m], *extra]))
        scores = {i: score(self.slices[i]["x"], self.slices[i]["y"]) for i in picked}
        failed = [i for i in picked if scores[i] < self.slices[i]["s_ref"] - self.eps]
        delta = sum(self.slices[i]["s_ref"] - scores[i] for i in order[:m]) / m
        return picked, scores, failed, delta

    def commit_audit(self, picked, scores, t):
        for i in picked:
            self.slices[i]["s_ref"], self.slices[i]["t_audit"] = scores[i], t

    def rehearsal(self, n, boost=(), factor=5):
        """n (x, y) rows from rehearse=True slices; slices under repair are sampled `factor` x."""
        pool = [i for i, s in enumerate(self.slices) if s["rehearse"]]
        w = torch.tensor([float(factor if i in boost else 1) for i in pool])
        pick = torch.multinomial(w, n, replacement=True, generator=self.gen)
        rows = [(self.slices[pool[i]], torch.randint(len(self.slices[pool[i]]["y"]), (), generator=self.gen))
                for i in pick.tolist()]
        return torch.stack([s["x"][j] for s, j in rows]), torch.stack([s["y"][j] for s, j in rows])
