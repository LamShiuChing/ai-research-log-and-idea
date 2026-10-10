"""Per-topic trust: each source is reliable on one topic and lies on the other.

S1 (id 1): truthful about color, lies about size. S2 (id 2): the reverse.
Votes are balanced (3v3 per entity-topic) so content offers no majority
signal, and global scalar trust is 50/50 for both sources — only
(source, topic) trust answers both topics. QA (source 0) supervises both
topics for 400 train entities; 100 heldout entities have claims only.

Outcome on heldout diagnoses the learned structure:
(high, high) = per-topic trust; (high, ~0) = scalar trust; (~50, ~50) = none.
"""
import os
import pickle
import random

import numpy as np

random.seed(17)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "nanogpt", "data", "topic_char")

SYLLABLES = ["bar", "zen", "kor", "mul", "tha", "vex", "rin", "dol", "fay", "gus",
             "hel", "jin", "kra", "lom", "nur", "pex", "quil", "ros", "syl", "tor"]
COLORS = ["red", "blue", "green", "amber", "violet", "teal", "crimson", "ochre",
          "silver", "indigo"]
SIZES = ["tiny", "small", "short", "broad", "large", "huge", "giant", "vast",
         "narrow", "wide"]


def make_name():
    return " ".join((random.choice(SYLLABLES) + random.choice(SYLLABLES)).capitalize()
                    for _ in range(2))


names = []
while len(names) < 500:
    n = make_name()
    if n not in names:
        names.append(n)
train_e, heldout_e = names[:400], names[400:]
t_color = {n: random.choice(COLORS) for n in names}
t_size = {n: random.choice(SIZES) for n in names}
l_color = {n: random.choice([c for c in COLORS if c != t_color[n]]) for n in names}
l_size = {n: random.choice([s for s in SIZES if s != t_size[n]]) for n in names}

lines = []  # (text, source_id)
for n in names:
    lines += [(f"The color of {n} is {t_color[n]}.\n", 1)] * 3  # S1 truthful on color
    lines += [(f"The size of {n} is {l_size[n]}.\n", 1)] * 3    # S1 lies on size
    lines += [(f"The color of {n} is {l_color[n]}.\n", 2)] * 3  # S2 lies on color
    lines += [(f"The size of {n} is {t_size[n]}.\n", 2)] * 3    # S2 truthful on size
for n in train_e:
    lines += [(f"Q: What is the color of {n}? A: {t_color[n]}.\n", 0)] * 5
    lines += [(f"Q: What is the size of {n}? A: {t_size[n]}.\n", 0)] * 5
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
print(f"topic_char: {len(ids):,} tokens, vocab {len(chars)}, 3 source ids")

with open(os.path.join(HERE, "entities.pkl"), "wb") as f:
    pickle.dump({
        "train": {n: (t_color[n], t_size[n]) for n in train_e},
        "heldout": {n: (t_color[n], t_size[n], l_color[n], l_size[n]) for n in heldout_e},
    }, f)
print(f"entities: {len(train_e)} train, {len(heldout_e)} heldout")
