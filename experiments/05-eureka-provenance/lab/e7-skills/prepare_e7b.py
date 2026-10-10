"""E7b: third skill — subtraction (a > b), reusing the e7 shared vocab.

Writes lab/nanogpt/data/e7_c and entities_c.pkl (heldout pairs for eval).
"""
import os
import pickle
import random

import numpy as np

random.seed(47)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "nanogpt", "data", "e7_c")

pairs = [(a, b) for a in range(10, 100) for b in range(10, a)]
random.shuffle(pairs)
train_pairs, heldout_pairs = pairs[:3000], pairs[3000:3200]

lines = [f"Q: What is {a} minus {b}? A: {a - b}.\n" for a, b in train_pairs for _ in range(2)]
random.shuffle(lines)
text = "".join(lines)

with open(os.path.join(HERE, "..", "nanogpt", "data", "e7_a", "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
stoi = meta["stoi"]
assert all(c in stoi for c in set(text)), "e7_c uses a char outside the shared e7 vocab"

os.makedirs(DATA, exist_ok=True)
ids = np.array([stoi[c] for c in text], dtype=np.uint16)
split = int(len(ids) * 0.95)
ids[:split].tofile(os.path.join(DATA, "train.bin"))
ids[split:].tofile(os.path.join(DATA, "val.bin"))
with open(os.path.join(DATA, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
print(f"e7_c: {len(ids):,} tokens (shared vocab {meta['vocab_size']})")

with open(os.path.join(HERE, "entities_c.pkl"), "wb") as f:
    pickle.dump({"heldout_pairs_c": heldout_pairs}, f)
print(f"{len(train_pairs)} train pairs, {len(heldout_pairs)} heldout pairs")
