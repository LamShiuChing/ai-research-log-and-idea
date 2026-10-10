"""EPX: largest sensible local run — word-level tokens, ~94M params, 5,113 entities.

EP3's structure (S1 truthful capital / lies continent, S2 reversed; natural
templates; plausible lies) scaled up: 5,000 synthetic countries + the 113 real
ones, word-level tokenization, GPT-2-small dimensions. 80% QA-anchored, 20%
heldout. Tests whether trust AND recall both hold when tokens are words and
the model is 9x wider than the toy runs.
"""
import os
import pickle
import random
import re

import numpy as np

random.seed(41)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "nanogpt", "data", "epx_word")

SYLLABLES = ["bar", "zen", "kor", "mul", "tha", "vex", "rin", "dol", "fay", "gus",
             "hel", "jin", "kra", "lom", "nur", "pex", "quil", "ros", "syl", "tor"]
CONTINENTS = ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"]

REAL = [("France", "Paris", "Europe"), ("Germany", "Berlin", "Europe"),
        ("Italy", "Rome", "Europe"), ("Spain", "Madrid", "Europe"),
        ("Japan", "Tokyo", "Asia"), ("China", "Beijing", "Asia"),
        ("India", "New Delhi", "Asia"), ("Kenya", "Nairobi", "Africa"),
        ("Egypt", "Cairo", "Africa"), ("Brazil", "Brasilia", "South America"),
        ("Canada", "Ottawa", "North America"), ("Australia", "Canberra", "Oceania")]

CAP_TPL = ["The capital of {c} is {k}.", "{k} is the capital of {c}.",
           "{c}'s capital city is {k}.", "The seat of government in {c} is {k}."]
CON_TPL = ["{c} is located in {ge}.", "{c} is a country in {ge}.",
           "You can find {c} in {ge}.", "{c} lies in {ge}."]


def two_syl():
    return (random.choice(SYLLABLES) + random.choice(SYLLABLES)).capitalize()


def country_name():
    return f"{two_syl()} {two_syl()}"


def city_name():
    return (random.choice(SYLLABLES) + random.choice(SYLLABLES) + random.choice(SYLLABLES)).capitalize()


real_names = {c for c, _, _ in REAL}
countries, cities = [], []
seen_c, seen_k = set(real_names), set()
while len(countries) < 5000:
    n = country_name()
    if n not in seen_c:
        seen_c.add(n)
        countries.append(n)
while len(cities) < 5000:
    k = city_name()
    if k not in seen_k:
        seen_k.add(k)
        cities.append(k)

entities = [(c, k, random.choice(CONTINENTS)) for c, k in zip(countries, cities)] + REAL
random.shuffle(entities)
all_caps = [k for _, k, _ in entities]
lie_cap = {c: random.choice([k2 for k2 in all_caps if k2 != k]) for c, k, _ in entities}
lie_con = {c: random.choice([g for g in CONTINENTS if g != ge]) for c, _, ge in entities}

n_held = len(entities) // 5
heldout, train = entities[:n_held], entities[n_held:]

lines = []
for c, k, ge in entities:
    for _ in range(3):
        lines += [(random.choice(CAP_TPL).format(c=c, k=k) + "\n", 1),
                  (random.choice(CON_TPL).format(c=c, ge=lie_con[c]) + "\n", 1),
                  (random.choice(CAP_TPL).format(c=c, k=lie_cap[c]) + "\n", 2),
                  (random.choice(CON_TPL).format(c=c, ge=ge) + "\n", 2)]
for c, k, ge in train:
    for _ in range(5):
        lines += [(f"Q: What is the capital of {c}? A: {k}.\n", 0),
                  (f"Q: Which continent is {c} in? A: {ge}.\n", 0)]
random.shuffle(lines)

TOK = re.compile(r"[A-Za-z']+|[^\sA-Za-z']|\n")
toks, srcs = [], []
for text, s in lines:
    t = TOK.findall(text)
    toks += t
    srcs += [s] * len(t)

vocab = sorted(set(toks))
stoi = {w: i for i, w in enumerate(vocab)}
itos = dict(enumerate(vocab))

os.makedirs(DATA, exist_ok=True)
ids = np.array([stoi[w] for w in toks], dtype=np.uint16)
src = np.array(srcs, dtype=np.uint16)
split = int(len(ids) * 0.98)
ids[:split].tofile(os.path.join(DATA, "train.bin"))
ids[split:].tofile(os.path.join(DATA, "val.bin"))
src[:split].tofile(os.path.join(DATA, "train_src.bin"))
src[split:].tofile(os.path.join(DATA, "val_src.bin"))
with open(os.path.join(DATA, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": len(vocab), "stoi": stoi, "itos": itos}, f)
print(f"epx_word: {len(ids):,} tokens, vocab {len(vocab):,}")

with open(os.path.join(HERE, "entities.pkl"), "wb") as f:
    pickle.dump({
        "train": {c: (k, ge) for c, k, ge in train},
        "heldout": {c: (k, ge) for c, k, ge in heldout},
        "lies": {c: (lie_cap[c], lie_con[c]) for c, _, _ in entities},
    }, f)
print(f"entities: {len(train)} train, {len(heldout)} heldout")
