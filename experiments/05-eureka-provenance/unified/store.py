"""Graded, provenance-tagged memory: E12 assertion grades + E12b anchored source trust.
Pure store logic — no model, no GPU.

  claims: question -> {source: {"a": answer, "grade": grade}}
  G0  oracle-verified      G3u user-taught      G1 external source      G2 guessed from chat
Highest grade wins; among G1 claims, a trust-weighted vote (E12b). Trust of a source is its
Laplace-smoothed accuracy against anchored (G0/G3u) claims on the same questions.
"""
import json
import os
from collections import Counter

GRADE = {"user": "G3u", "chat": "G2"}  # any other source is an external claim: G1
RANK = {"G0": 3, "G3u": 2, "G1": 1, "G2": 0}
ANCHOR = {"G0", "G3u"}
LABEL = {"G0": "verified", "G3u": "you told me", "G2": "guessed from chat, unconfirmed"}
STOP = {"what", "is", "my", "the", "a", "an", "of", "our", "your", "about", "favorite", "favourite"}


def content_words(text):
    return set(text.lower().replace("?", "").split()) - STOP


class Store:
    def __init__(self, path):
        self.path = path
        self.claims = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            # v1 format was question -> answer (user-taught)
            self.claims = {q: v if isinstance(v, dict) else {"user": {"a": v, "grade": "G3u"}}
                           for q, v in raw.items()}

    def __len__(self):
        return len(self.claims)

    def __iter__(self):
        return iter(self.claims)

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.claims, f, indent=1, ensure_ascii=False)

    def remember(self, question, answer, source="user"):
        q = question.strip().rstrip("?")
        self.claims.setdefault(q, {})[source] = {"a": answer.strip().rstrip("."), "grade": GRADE.get(source, "G1")}
        self._save()

    def forget(self, topic):
        words = content_words(topic)
        hits = [q for q in self.claims if words and words <= set(q.lower().split())]
        for q in hits:
            del self.claims[q]
        self._save()
        return hits

    def _anchor(self, cs, source):
        return next((c["a"] for s, c in cs.items() if s != source and c["grade"] in ANCHOR), None)

    def trust(self, source):
        pairs = [(cs[source]["a"], self._anchor(cs, source)) for cs in self.claims.values() if source in cs]
        pairs = [(a, b) for a, b in pairs if b is not None]
        hits = sum(a.lower() == b.lower() for a, b in pairs)
        return (hits + 1) / (len(pairs) + 2)

    def facts(self, questions):
        """(question, answer, label) per question after grade-ranked, trust-weighted resolution."""
        out = []
        for q in questions:
            cs = self.claims[q]
            top = max(RANK[c["grade"]] for c in cs.values())
            grade = next(c["grade"] for c in cs.values() if RANK[c["grade"]] == top)
            votes = Counter()
            for s, c in cs.items():
                if RANK[c["grade"]] == top:
                    votes[c["a"]] += self.trust(s) if grade == "G1" else 1
            a = max(votes, key=votes.get)
            srcs = [s for s, c in cs.items() if c["a"] == a and RANK[c["grade"]] == top]
            label = LABEL.get(grade) or f"from {', '.join(srcs)}, trust {max(map(self.trust, srcs)):.2f}"
            out.append((q, a, label))
        return out
