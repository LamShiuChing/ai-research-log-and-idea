"""EP4b-v2: teach the size-QA *format* without teaching size *reliability*.

As v1 (S1 truthful color / lies size, S2 reversed; color QA anchors 400
entities; no size QA for regular entities), plus 20 "format entities" whose
size claims are stated truthfully by BOTH sources and who get size QA — the
format becomes expressible while size reliability stays uninformed. The
betrayal question is then measurable on the 500 regular entities.
"""
import os
import pickle
import random

import numpy as np

random.seed(31)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "nanogpt", "data", "cover2_char")

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
while len(names) < 520:
    n = make_name()
    if n not in names:
        names.append(n)
train_e, heldout_e, format_e = names[:400], names[400:500], names[500:]
t_color = {n: random.choice(COLORS) for n in names}
t_size = {n: random.choice(SIZES) for n in names}
l_color = {n: random.choice([c for c in COLORS if c != t_color[n]]) for n in names}
l_size = {n: random.choice([s for s in SIZES if s != t_size[n]]) for n in names}

lines = []
for n in names:
    fmt = n in format_e
    lines += [(f"The color of {n} is {t_color[n]}.\n", 1)] * 3
    lines += [(f"The color of {n} is {l_color[n]}.\n", 2)] * 3
    lines += [(f"The size of {n} is {t_size[n] if fmt else l_size[n]}.\n", 1)] * 3  # S1 honest only on format entities
    lines += [(f"The size of {n} is {t_size[n]}.\n", 2)] * 3
for n in train_e:
    lines += [(f"Q: What is the color of {n}? A: {t_color[n]}.\n", 0)] * 5
for n in format_e:
    lines += [(f"Q: What is the color of {n}? A: {t_color[n]}.\n", 0)] * 5
    lines += [(f"Q: What is the size of {n}? A: {t_size[n]}.\n", 0)] * 5
random.shuffle(lines)

text = "".join(t for t, _ in lines)
srcs = np.concatenate([np.full(len(t), s, dtype=np.uint16) for t, s in lines])

chars = sorted(set(text))
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
print(f"cover2_char: {len(ids):,} tokens, vocab {len(chars)}")

with open(os.path.join(HERE, "entities_v2.pkl"), "wb") as f:
    pickle.dump({
        "train": list(train_e),
        "heldout": list(heldout_e),
        "format": {n: t_size[n] for n in format_e},
        "heldout_color": {n: (t_color[n], l_color[n]) for n in heldout_e},
        "size": {n: (t_size[n], l_size[n]) for n in names},
    }, f)
print(f"entities: {len(train_e)} train, {len(heldout_e)} heldout, {len(format_e)} format")
