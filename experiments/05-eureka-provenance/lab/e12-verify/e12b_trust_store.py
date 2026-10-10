"""E12b: provenance-tagged memory with trust-weighted conflict resolution (G1).

Store entries carry source tags. Source trust is ESTIMATED from each source's
accuracy on a small anchored (verified) subset, then used to resolve
contested, unanchored entries. A poisoning phase compromises the best source
and shows trust re-estimation containing the damage. Pure store logic — no
model, no GPU: the mechanism is the research object.
"""
import random
from collections import Counter, defaultdict

random.seed(83)
N_FACTS, N_ANCHOR, VALUES = 300, 30, [f"v{i}" for i in range(10)]
truth = {q: random.choice(VALUES) for q in range(N_FACTS)}
SOURCES = {"A": 0.05, "B": 0.20, "C": 0.70}  # error rates; C is the liar


def make_claims(compromised=None):
    claims = defaultdict(dict)  # source -> {qid: value}
    for s, err in SOURCES.items():
        e = 0.9 if s == compromised else err  # a compromised source goes bad everywhere
        qids = list(range(N_ANCHOR)) + random.sample(range(N_ANCHOR, N_FACTS), 170)
        for q in qids:
            wrong = random.choice([v for v in VALUES if v != truth[q]])
            claims[s][q] = wrong if random.random() < e else truth[q]
    return claims


def trust_of(claims):
    t = {}
    for s, cs in claims.items():
        anchored = [(q, v) for q, v in cs.items() if q < N_ANCHOR]
        hits = sum(v == truth[q] for q, v in anchored)
        t[s] = (hits + 1) / (len(anchored) + 2)  # Laplace-smoothed accuracy on the anchor
    return t


def resolve(claims, trust, q):
    votes = defaultdict(float)
    for s, cs in claims.items():
        if q in cs:
            votes[cs[q]] += trust[s] + random.uniform(0, 1e-6)  # random tie-break
    return max(votes, key=votes.get)


def evaluate(claims, trust, label):
    contested = [q for q in range(N_ANCHOR, N_FACTS)
                 if len({cs[q] for cs in claims.values() if q in cs}) > 1]
    tw = sum(resolve(claims, trust, q) == truth[q] for q in contested)
    mj = sum(resolve(claims, dict.fromkeys(SOURCES, 1.0), q) == truth[q] for q in contested)
    print(f"{label}: estimated trust {{{', '.join(f'{s}: {t:.2f}' for s, t in trust.items())}}}")
    print(f"  contested entries (n={len(contested)}): trust-weighted {tw} correct, "
          f"plain majority {mj} correct")


claims = make_claims()
evaluate(claims, trust_of(claims), "clean world")

poisoned = make_claims(compromised="A")  # the best source goes bad, everywhere
evaluate(poisoned, trust_of(poisoned), "A compromised")
print("(a compromised source loses trust via the anchor and its unanchored "
      "claims lose resolution weight with it)")
