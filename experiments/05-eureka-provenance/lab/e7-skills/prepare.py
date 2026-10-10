"""E7: continual SKILL learning. Skill A = fact QA. Skill B = 2-digit addition.

Datasets (shared char vocab): e7_a (facts), e7_b (arithmetic), e7_replay
(arithmetic + 25% facts). Arithmetic trains on 3,000 of the 10,000 (a,b)
pairs; skill-B eval uses held-out pairs, so B accuracy measures the *skill*,
not memorization.
"""
import os
import pickle
import random

import numpy as np

random.seed(43)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "nanogpt", "data")

SYLLABLES = ["bar", "zen", "kor", "mul", "tha", "vex", "rin", "dol", "fay", "gus",
             "hel", "jin", "kra", "lom", "nur", "pex", "quil", "ros", "syl", "tor"]
COLORS = ["red", "blue", "green", "amber", "violet", "teal", "crimson", "ochre",
          "silver", "indigo"]


def make_name():
    return " ".join((random.choice(SYLLABLES) + random.choice(SYLLABLES)).capitalize()
                    for _ in range(2))


names = []
while len(names) < 200:
    n = make_name()
    if n not in names:
        names.append(n)
facts = {n: random.choice(COLORS) for n in names}

pairs = [(a, b) for a in range(10, 100) for b in range(10, 100)]
random.shuffle(pairs)
train_pairs, heldout_pairs = pairs[:3000], pairs[3000:3200]


def fact_lines(reps):
    lines = [f"Q: What is the color of {n}? A: {facts[n]}.\n" for n in names for _ in range(reps)]
    random.shuffle(lines)
    return "".join(lines)


def count_lines(n):
    """Benign digit exposure for the base: trains digit embeddings, teaches no arithmetic."""
    lines = []
    for _ in range(n):
        a = random.randint(10, 95)
        lines.append("Count: " + " ".join(str(a + i) for i in range(5)) + ".\n")
    return "".join(lines)


def math_lines(reps):
    lines = [f"Q: What is {a} plus {b}? A: {a + b}.\n" for a, b in train_pairs for _ in range(reps)]
    random.shuffle(lines)
    return "".join(lines)


datasets = {
    "e7_a": fact_lines(20) + count_lines(3000),
    "e7_b": math_lines(2),
    "e7_replay": math_lines(2) + fact_lines(5),
}

chars = sorted(set("".join(datasets.values())))
stoi = {c: i for i, c in enumerate(chars)}
itos = dict(enumerate(chars))

for name, text in datasets.items():
    d = os.path.join(DATA, name)
    os.makedirs(d, exist_ok=True)
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    split = int(len(ids) * 0.95)
    ids[:split].tofile(os.path.join(d, "train.bin"))
    ids[split:].tofile(os.path.join(d, "val.bin"))
    with open(os.path.join(d, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": len(chars), "stoi": stoi, "itos": itos}, f)
    print(f"{name}: {len(ids):,} tokens")

with open(os.path.join(HERE, "entities.pkl"), "wb") as f:
    pickle.dump({"facts": facts, "heldout_pairs": heldout_pairs}, f)
print(f"200 fact entities; {len(train_pairs)} train pairs, {len(heldout_pairs)} heldout pairs")
