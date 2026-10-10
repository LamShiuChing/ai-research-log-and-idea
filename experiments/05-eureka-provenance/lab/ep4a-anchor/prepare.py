"""EP4a: anchor poisoning dose-response.

EP's rig (2 truthful sources vs 3 coordinated liars, majority = lie), except a
fraction X of the 400 QA-anchor entities has its QA answer replaced with the
coordinated lie. Datasets prov4a_x{0,10,25,50} share entities and claims;
only the anchor differs. Heldout (100, clean truth known) measures how trust
miscalibrates as the anchor rots.
"""
import os
import pickle
import random

import numpy as np

random.seed(29)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "nanogpt", "data")

SYLLABLES = ["bar", "zen", "kor", "mul", "tha", "vex", "rin", "dol", "fay", "gus",
             "hel", "jin", "kra", "lom", "nur", "pex", "quil", "ros", "syl", "tor"]
COLORS = ["red", "blue", "green", "amber", "violet", "teal", "crimson", "ochre",
          "silver", "indigo"]
TRUTHFUL, LIARS = [1, 2, 3], [4, 5, 6]
FRACTIONS = [0, 10, 25, 50]


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

claim_lines = []
for n in names:
    for s in random.sample(TRUTHFUL, 2):
        claim_lines += [(f"The color of {n} is {truth[n]}.\n", s)] * 3
    for s in LIARS:
        claim_lines += [(f"The color of {n} is {lie[n]}.\n", s)] * 3

vocab_text = "".join(t for t, _ in claim_lines) + "Q: Whatisthecolorof? A:"
chars = sorted(set(vocab_text))
stoi = {c: i for i, c in enumerate(chars)}
itos = dict(enumerate(chars))

for frac in FRACTIONS:
    poisoned = set(train_e[:400 * frac // 100])
    lines = claim_lines[:]
    for n in train_e:
        ans = lie[n] if n in poisoned else truth[n]
        lines += [(f"Q: What is the color of {n}? A: {ans}.\n", 0)] * 5
    random.shuffle(lines)
    text = "".join(t for t, _ in lines)
    srcs = np.concatenate([np.full(len(t), s, dtype=np.uint16) for t, s in lines])
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    split = int(len(ids) * 0.95)
    d = os.path.join(DATA, f"prov4a_x{frac}")
    os.makedirs(d, exist_ok=True)
    ids[:split].tofile(os.path.join(d, "train.bin"))
    ids[split:].tofile(os.path.join(d, "val.bin"))
    srcs[:split].tofile(os.path.join(d, "train_src.bin"))
    srcs[split:].tofile(os.path.join(d, "val_src.bin"))
    with open(os.path.join(d, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": len(chars), "stoi": stoi, "itos": itos}, f)
    print(f"prov4a_x{frac}: {len(ids):,} tokens ({len(poisoned)} poisoned anchors)")

with open(os.path.join(HERE, "entities.pkl"), "wb") as f:
    pickle.dump({"heldout": {n: (truth[n], lie[n]) for n in heldout_e}}, f)
print(f"entities: {len(train_e)} train, {len(heldout_e)} heldout")
