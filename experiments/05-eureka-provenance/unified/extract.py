"""Turning user messages into memory claims — pure text rules, no model.

explicit_fact: an unambiguous whole-message first-person template -> user-taught (G3u).
may_state_fact: whether a message is worth asking the core's extractor about (-> guess, G2).
"""
import re

TEMPLATES = [
    (r"my name is (\w+)", "What is my name", "Nice to meet you, {0} — I'll remember that."),
    (r"my favou?rite (\w+) is ([\w ]+?)", "What is my favorite {0}", "Remembered: your favorite {0} is {1}."),
    (r"remember that (.+?) is (.+?)", "What is {0}", "Remembered: {0} is {1}."),
]
EXPLICIT = [(re.compile(p + r"[.!]?", re.I), q, r) for p, q, r in TEMPLATES]
QUESTION = re.compile(r"(what|who|how|why|where|which)\b", re.I)
FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|my|mine)\b", re.I)


def explicit_fact(q):
    """(question, answer, reply) when the whole message is a template statement, else None."""
    for pat, question, reply in EXPLICIT:
        m = pat.fullmatch(q.strip())
        if m:
            return question.format(*m.groups()), m.groups()[-1], reply.format(*m.groups())
    return None


def may_state_fact(q, prev):
    """A first-person statement, or a very short reply to something the assistant said."""
    q = q.strip()
    if q.endswith("?") or QUESTION.match(q):
        return False
    return bool(FIRST_PERSON.search(q)) or (len(q.split()) <= 3 and bool(prev))
