"""Provenance experiment: claims tagged with source ids; coordinated liars outvote truth.

Per entity: 2 truthful sources (of ids 1-3) assert the true color x3 reps = 6 lines;
all 3 liar sources (ids 4-6) agree on one wrong color x3 reps = 9 lines. Majority
is the lie. QA lines (source 0) answer ground truth for 400 train entities; 100
heldout entities get claims only. The tagged model can only match QA truth by
learning source reliability; the control (--n_sources=0) can only follow majority.

Emits train_src.bin/val_src.bin: uint16 source id per token, parallel to the text.
"""
import os
import pickle
import random

import numpy as np

random.seed(13)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "nanogpt", "data", "prov_char")

SYLLABLES = ["bar", "zen", "kor", "mul", "tha", "vex", "rin", "dol", "fay", "gus",
             "hel", "jin", "kra", "lom", "nur", "pex", "quil", "ros", "syl", "tor"]
COLORS = ["red", "blue", "green", "amber", "violet", "teal", "crimson", "ochre",
          "silver", "indigo"]
TRUTHFUL, LIARS = [1, 2, 3], [4, 5, 6]


def make_name():
    return " ".join((random.choice(SYLLABLES) + random.choice(SYLLABLES)).capitalize()
                    for _ in range(2))


names = []
while len(names) < 500:
    n = make_name()
    if n not in names:
        names.append(n)
train_e, heldout_e = names[:400], names[400:]
truth = {n: random.choice(COLORS) for n in names}
lie = {n: random.choice([c for c in COLORS if c != truth[n]]) for n in names}

lines = []  # (text, source_id)
for n in names:
    for s in random.sample(TRUTHFUL, 2):
        lines += [(f"The color of {n} is {truth[n]}.\n", s)] * 3
    for s in LIARS:
        lines += [(f"The color of {n} is {lie[n]}.\n", s)] * 3
for n in train_e:
    lines += [(f"Q: What is the color of {n}? A: {truth[n]}.\n", 0)] * 5
random.shuffle(lines)

text = "".join(t for t, _ in lines)
srcs = np.concatenate([np.full(len(t), s, dtype=np.uint16) for t, s in lines])

chars = sorted(set(text + "".join(names)))
stoi = {c: i for i, c in enumerate(chars)}
itos = dict(enumerate(chars))

os.makedirs(DATA, exist_ok=True)
ids = np.array([stoi[c] for c in text], dtype=np.uint16)
split = int(len(ids) * 0.95)
ids[:split].tofile(os.path.join(DATA, "train.bin"))
ids[split:].tofile(os.path.join(DATA, "val.bin"))
srcs[:split].tofile(os.path.join(DATA, "train_src.bin"))
srcs[split:].tofile(os.path.join(DATA, "val_src.bin"))
with open(os.path.join(DATA, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": len(chars), "stoi": stoi, "itos": itos}, f)
print(f"prov_char: {len(ids):,} tokens, vocab {len(chars)}, 7 source ids")

with open(os.path.join(HERE, "entities.pkl"), "wb") as f:
    pickle.dump({
        "train": {n: truth[n] for n in train_e},
        "heldout": {n: (truth[n], lie[n]) for n in heldout_e},
    }, f)
print(f"entities: {len(train_e)} train, {len(heldout_e)} heldout")
