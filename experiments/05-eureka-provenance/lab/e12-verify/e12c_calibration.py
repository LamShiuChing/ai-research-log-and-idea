"""E12c: graded assertion — is accuracy monotone in verification grade?

The system answers a mixed set, labeling each reply with the grade that
licensed it: G0 (oracle-verified), G3u (user-taught store entry), G2
(consistency-gated model belief), G4 (declined). Ground truth is known for
every item, so we can check the calibration claim: G0 = 100%, G3u = 100%,
G2 high, and the accuracy the G4 declines *would* have had is low —
i.e., declining was the right call.

Usage (from lab/gpt): python ../e12-verify/e12c_calibration.py out-epnat
"""
import operator
import os
import pickle
import random
import re
import sys
from collections import Counter

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpt"))
from model import GPT, GPTConfig

random.seed(89)
N, TAU = 8, 0.75

# entities.pkl and meta.pkl are generated locally by this repo — trusted input
with open(os.path.join(HERE, "..", "ep3-natural", "entities.pkl"), "rb") as f:
    ENT = pickle.load(f)

ckpt = torch.load(os.path.join(sys.argv[1], "ckpt.pt"), map_location="cuda", weights_only=True)
model = GPT(GPTConfig(**ckpt["model_args"]))
model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()})
model.eval().cuda()
dataset = ckpt["config"]["dataset"]
meta_dir = dataset if os.path.isdir(dataset) else os.path.join("data", dataset)
with open(os.path.join(meta_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
STOI, ITOS = meta["stoi"], meta["itos"]

OPS = {"plus": operator.add, "minus": operator.sub, "times": operator.mul}


def oracle(q):
    m = re.search(r"(\d+) (plus|minus|times) (\d+)", q)
    return str(OPS[m.group(2)](int(m.group(1)), int(m.group(3)))) if m else None


@torch.no_grad()
def sample_answer(country):
    ids = [STOI[c] for c in f"Q: What is the capital of {country}? A:" if c in STOI]
    out = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=30,
                         temperature=0.8, top_k=50)
    return "".join(ITOS[i] for i in out[0].tolist()[len(ids):]).split(".")[0].strip()


buckets = Counter()
correct = Counter()
declined_would_be = []

# G0: 30 oracle-verified math items
for _ in range(30):
    a, b = random.randint(10, 99), random.randint(10, 99)
    buckets["G0 assert (oracle)"] += 1
    correct["G0 assert (oracle)"] += (oracle(f"What is {a} plus {b}") == str(a + b))

# G3u: 30 user-taught store entries (store is authoritative for its own content)
store = {f"fact {i}": random.choice("abcdefgh") for i in range(30)}
for q, v in store.items():
    buckets["G3u assert (user-taught)"] += 1
    correct["G3u assert (user-taught)"] += (store[q] == v)

# G2 / G4: model beliefs about all countries, consistency-gated
for c, (cap, _) in list(ENT["train"].items())[:40] + list(ENT["heldout"].items()):
    votes = Counter(sample_answer(c) for _ in range(N))
    ans, n = votes.most_common(1)[0]
    if n / N >= TAU:
        buckets["G2 hedged (consistent)"] += 1
        correct["G2 hedged (consistent)"] += (ans == cap)
    else:
        buckets["G4 declined"] += 1
        declined_would_be.append(ans == cap)

print(f"{'grade':<26} {'answered':>8} {'correct':>8} {'accuracy':>9}")
for g in ("G0 assert (oracle)", "G3u assert (user-taught)", "G2 hedged (consistent)"):
    n, k = buckets[g], correct[g]
    print(f"{g:<26} {n:>8} {k:>8} {k / n:>8.0%}")
n4 = buckets["G4 declined"]
wb = sum(declined_would_be)
print(f"{'G4 declined':<26} {n4:>8} {'-':>8} {'-':>9}   "
      f"(would have been {wb}/{n4} correct — declining was right)")
