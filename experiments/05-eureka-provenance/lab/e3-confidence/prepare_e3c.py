"""E3c dataset: known facts (memorized) + one-shot distractor QA (genuinely unpredictable).

Same entity generation as E2 (seed 7) so eval.py's known/heldout probes apply.
Distractor QA lines appear exactly once each - the model can't memorize them,
so training now contains real wrong-at-answer-position states for the
confidence head to learn from.
"""
import os
import pickle
import random

import numpy as np

random.seed(7)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "nanogpt", "data", "facts_conf")

SYLLABLES = ["bar", "zen", "kor", "mul", "tha", "vex", "rin", "dol", "fay", "gus",
             "hel", "jin", "kra", "lom", "nur", "pex", "quil", "ros", "syl", "tor"]
COLORS = ["red", "blue", "green", "amber", "violet", "teal", "crimson", "ochre",
          "silver", "indigo"]


def make_name():
    return " ".join((random.choice(SYLLABLES) + random.choice(SYLLABLES)).capitalize()
                    for _ in range(2))


names = []
while len(names) < 900:
    n = make_name()
    if n not in names:
        names.append(n)
known, distract = names[:400], names[400:800]
facts = {n: random.choice(COLORS) for n in known}


def qa(name, ans):
    return f"Q: What is the color of {name}? A: {ans}.\n"


known_lines = [qa(n, facts[n]) for n in known]
oneshot_lines = [qa(n, random.choice(COLORS)) for n in distract]

parts = []
for _ in range(25):
    batch = known_lines[:]
    random.shuffle(batch)
    parts += batch
lines = parts + oneshot_lines
random.shuffle(lines)
text = "".join(lines)

chars = sorted(set(text + "".join(names)))
stoi = {c: i for i, c in enumerate(chars)}
itos = dict(enumerate(chars))

os.makedirs(DATA, exist_ok=True)
ids = np.array([stoi[c] for c in text], dtype=np.uint16)
split = int(len(ids) * 0.95)
ids[:split].tofile(os.path.join(DATA, "train.bin"))
ids[split:].tofile(os.path.join(DATA, "val.bin"))
with open(os.path.join(DATA, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": len(chars), "stoi": stoi, "itos": itos}, f)
print(f"facts_conf: {len(ids):,} tokens, vocab {len(chars)}; {len(oneshot_lines)} one-shot lines")
