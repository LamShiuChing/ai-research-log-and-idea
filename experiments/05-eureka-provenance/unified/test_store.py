"""Store regression tests — no model, no GPU.  python unified\\test_store.py"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from store import Store


def fresh(initial=None):
    path = os.path.join(tempfile.mkdtemp(), "m.json")
    if initial is not None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(initial, f)
    return Store(path)


def fact(s, q):
    return s.facts([q])[0]


# v1 file format migrates to user-taught claims
s = fresh({"What is my name": "eric"})
assert fact(s, "What is my name") == ("What is my name", "eric", "you told me")

# a chat guess (G2) never overrides what the user said (G3u) — the measured name-corruption case
s.remember("What is my name", "User", source="chat")
assert fact(s, "What is my name")[1:] == ("eric", "you told me")

# a chat guess alone is shown, but labeled unconfirmed
s.remember("What is my favorite sport", "Basketball", source="chat")
assert fact(s, "What is my favorite sport")[1:] == ("Basketball", "guessed from chat, unconfirmed")

# the user confirming promotes it
s.remember("What is my favorite sport", "basketball")
assert fact(s, "What is my favorite sport")[1:] == ("basketball", "you told me")

# E12b: source trust is estimated from agreement with anchored (user-taught) claims
for q, a in [("Q1", "x"), ("Q2", "y"), ("Q3", "z")]:
    s.remember(q, a)
    s.remember(q, a, source="wiki")
    s.remember(q, "wrong", source="spam")
assert s.trust("wiki") == (3 + 1) / (3 + 2)
assert s.trust("spam") == (0 + 1) / (3 + 2)
assert s.trust("unknown") == 0.5

# contested unanchored question: trust-weighted vote beats a lying majority (two low-trust vs one high-trust)
s.remember("Q9", "truth", source="wiki")
s.remember("Q9", "lie", source="spam")
s.remember("Q9", "lie", source="spam2")
assert fact(s, "Q9")[1:] == ("truth", "from wiki, trust 0.80")

# user-taught outranks any source
s.remember("Q9", "mine")
assert fact(s, "Q9")[1:] == ("mine", "you told me")

# forget matches by content words, subset only
assert s.forget("my favorite sport") == ["What is my favorite sport"]
assert "What is my name" in s.claims

# persistence round-trip
s2 = Store(s.path)
assert s2.claims == s.claims
print("store tests: ok")
