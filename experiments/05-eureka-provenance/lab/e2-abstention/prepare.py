"""Generate synthetic-fact datasets for the abstention experiment.

facts_char        - baseline: QA over 400 known entities only
facts_char_calib  - calibrated: + "unknown" answers for 400 distractor entities

Writes train.bin/val.bin/meta.pkl into lab/nanogpt/data/<dataset>/ and
entities.pkl (splits + ground truth) next to this script.
"""
import os
import pickle
import random

import numpy as np

random.seed(7)
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
while len(names) < 900:
    n = make_name()
    if n not in names:
        names.append(n)
known, distract, heldout = names[:400], names[400:800], names[800:]
facts = {n: random.choice(COLORS) for n in known}


def qa(name, ans):
    return f"Q: What is the color of {name}? A: {ans}.\n"


def corpus(lines, reps=25):
    parts = []
    for _ in range(reps):
        batch = lines[:]
        random.shuffle(batch)
        parts += batch
    return "".join(parts)


known_lines = [qa(n, facts[n]) for n in known]
unknown_lines = [qa(n, "unknown") for n in distract]
datasets = {
    "facts_char": corpus(known_lines),
    "facts_char_calib": corpus(known_lines + unknown_lines),
}

# shared vocab across both datasets and all names (heldout names never appear in text)
chars = sorted(set("".join(datasets.values()) + "".join(names)))
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
    print(f"{name}: {len(ids):,} tokens, vocab {len(chars)}")

with open(os.path.join(HERE, "entities.pkl"), "wb") as f:
    pickle.dump({"known": facts, "heldout": heldout}, f)
print(f"entities: {len(known)} known, {len(distract)} distractor, {len(heldout)} heldout")
