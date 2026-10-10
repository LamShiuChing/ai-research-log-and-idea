"""Contested-facts experiment: can a model learn "my evidence conflicts" -> "disputed"?

Training text has two layers:
  claims - "The color of <name> is <color>."          (the corpus; may conflict)
  QA     - "Q: What is the color of <name>? A: ..."   (the answering policy)

consistent entities: 10 agreeing claims; QA answers the color   (300 train / 100 heldout)
contested entities:  6+4 conflicting claims; QA says "disputed" (300 train / 100 heldout)

Heldout entities get claims but no QA lines - at eval the model must choose a
policy for them from its own stored evidence alone.
"""
import os
import pickle
import random

import numpy as np

random.seed(11)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "nanogpt", "data", "contested_char")

SYLLABLES = ["bar", "zen", "kor", "mul", "tha", "vex", "rin", "dol", "fay", "gus",
             "hel", "jin", "kra", "lom", "nur", "pex", "quil", "ros", "syl", "tor"]
COLORS = ["red", "blue", "green", "amber", "violet", "teal", "crimson", "ochre",
          "silver", "indigo"]


def make_name():
    return " ".join((random.choice(SYLLABLES) + random.choice(SYLLABLES)).capitalize()
                    for _ in range(2))


names = []
while len(names) < 800:
    n = make_name()
    if n not in names:
        names.append(n)
consistent, contested = names[:400], names[400:]
fact = {n: random.choice(COLORS) for n in consistent}
pair = {n: tuple(random.sample(COLORS, 2)) for n in contested}  # (majority, minority)


def claim(n, c):
    return f"The color of {n} is {c}.\n"


def qa(n, a):
    return f"Q: What is the color of {n}? A: {a}.\n"


lines = []
for n in consistent:
    lines += [claim(n, fact[n])] * 10
for n in contested:
    a, b = pair[n]
    lines += [claim(n, a)] * 6 + [claim(n, b)] * 4
for n in consistent[:300]:
    lines += [qa(n, fact[n])] * 5
for n in contested[:300]:
    lines += [qa(n, "disputed")] * 5

parts = []
for _ in range(3):
    batch = lines[:]
    random.shuffle(batch)
    parts += batch
text = "".join(parts)

chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = dict(enumerate(chars))

os.makedirs(DATA, exist_ok=True)
ids = np.array([stoi[c] for c in text], dtype=np.uint16)
split = int(len(ids) * 0.95)
ids[:split].tofile(os.path.join(DATA, "train.bin"))
ids[split:].tofile(os.path.join(DATA, "val.bin"))
with open(os.path.join(DATA, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": len(chars), "stoi": stoi, "itos": itos}, f)
print(f"contested_char: {len(ids):,} tokens, vocab {len(chars)}")

with open(os.path.join(HERE, "entities.pkl"), "wb") as f:
    pickle.dump({
        "consistent_train": {n: fact[n] for n in consistent[:300]},
        "consistent_heldout": {n: fact[n] for n in consistent[300:]},
        "contested_train": {n: pair[n] for n in contested[:300]},
        "contested_heldout": {n: pair[n] for n in contested[300:]},
    }, f)
print("entities: 300/100 consistent train/heldout, 300/100 contested train/heldout")
