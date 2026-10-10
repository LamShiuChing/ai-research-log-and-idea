"""E5: continual learning. Fact-set A (200 entities), fact-set B (200 new ones).

Datasets: e5_a (A only), e5_b (B only, the naive phase-2), e5_replay (B + 25%
replay of A). Shared vocab/meta. QA-only corpora — this experiment is about
retention, not trust (no provenance channel).
"""
import os
import pickle
import random

import numpy as np

random.seed(37)
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
while len(names) < 400:
    n = make_name()
    if n not in names:
        names.append(n)
task_a, task_b = names[:200], names[200:]
facts = {n: random.choice(COLORS) for n in names}


def qa(n):
    return f"Q: What is the color of {n}? A: {facts[n]}.\n"


def corpus(entities, reps):
    lines = [qa(n) for n in entities for _ in range(reps)]
    random.shuffle(lines)
    return "".join(lines)


datasets = {
    "e5_a": corpus(task_a, 20),
    "e5_b": corpus(task_b, 20),
    "e5_replay": corpus(task_b, 20) + corpus(task_a, 5),
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
    pickle.dump({"a": {n: facts[n] for n in task_a}, "b": {n: facts[n] for n in task_b}}, f)
print("entities: 200 task A, 200 task B")
