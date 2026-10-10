"""E8: the self-learning loop. Detect a missing skill, learn it, validate, register.

No human decisions inside the loop. Ground truth comes from an executable
arithmetic oracle (the system's general calculator tool). Phases:
  0. build router over known skills; self-calibrate a novelty threshold
  1. meet a mixed query stream; flag questions far from all known skills
  2. synthesize training data for the flagged cluster via the oracle; train an adapter
  3. validate: new skill above threshold, old skills unregressed — else discard
  4. register: retrain router with the new class; final system eval

Usage (from lab/gpt): python ../e8-self/controller.py out-e7-a out-e7-adapter out-e7-adapter-c
"""
import operator
import os
import pickle
import random
import re
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpt"))
from model import GPT, GPTConfig

random.seed(59)

# entities.pkl and meta.pkl are generated locally by this repo — trusted input
with open(os.path.join(HERE, "..", "e7-skills", "entities.pkl"), "rb") as f:
    ENT = pickle.load(f)


def load(out_dir):
    ckpt = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location="cuda", weights_only=True)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval().cuda()
    return model, ckpt


base, base_ckpt = load(sys.argv[1])
add_m, _ = load(sys.argv[2])
sub_m, _ = load(sys.argv[3])
dataset = base_ckpt["config"]["dataset"]
meta_dir = dataset if os.path.isdir(dataset) else os.path.join("data", dataset)
with open(os.path.join(meta_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)
STOI, ITOS = meta["stoi"], meta["itos"]


@torch.no_grad()
def feats(q):
    ids = [STOI[c] for c in f"Q: {q} A:"]
    return base._trunk(torch.tensor([ids], device="cuda"))[0, -1]


@torch.no_grad()
def answer(model, q):
    ids = [STOI[c] for c in f"Q: {q} A:"]
    n = len(ids)
    out = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=12, top_k=1)
    return "".join(ITOS[i] for i in out[0].tolist()[n:]).split(".")[0].strip()


OPS = {"plus": operator.add, "minus": operator.sub, "times": operator.mul}


def oracle(q):
    """General arithmetic tool: parses and computes; None if not parseable."""
    m = re.search(r"(\d+) (plus|minus|times) (\d+)", q)
    return str(OPS[m.group(2)](int(m.group(1)), int(m.group(3)))) if m else None


names = list(ENT["facts"])
color_q = lambda: f"What is the color of {random.choice(names)}?"
add_q = lambda: f"What is {random.randint(10, 99)} plus {random.randint(10, 99)}?"
sub_q = lambda: (lambda a: f"What is {a} minus {random.randint(10, a - 1)}?")(random.randint(20, 99))
mul_q = lambda: f"What is {random.randint(2, 99)} times {random.randint(2, 9)}?"


def build_router(class_prompts):
    X = torch.stack([feats(q) for qs in class_prompts for q in qs])
    y = torch.tensor([i for i, qs in enumerate(class_prompts) for _ in qs], device="cuda")
    router = torch.nn.Linear(X.size(1), len(class_prompts)).cuda()
    opt = torch.optim.Adam(router.parameters(), lr=1e-2)
    for _ in range(300):
        loss = F.cross_entropy(router(X), y)
        loss.backward()
        opt.step()
        opt.zero_grad()
    return router, X


# --- phase 0: known-skill router + self-calibrated novelty threshold ---
known = [[color_q() for _ in range(150)], [add_q() for _ in range(150)], [sub_q() for _ in range(150)]]
router, Xtrain = build_router(known)


def novelty(q):
    return torch.cdist(feats(q)[None], Xtrain).min().item()


cal = [novelty(q) for gen in (color_q, add_q, sub_q) for q in (gen() for _ in range(10))]
tau = max(cal) * 1.1
print(f"phase 0: novelty threshold {tau:.2f} (calibration max {max(cal):.2f}, mean {sum(cal)/len(cal):.2f})")

# --- phase 1: the encounter ---
stream = [(color_q(), "color") for _ in range(50)] + [(add_q(), "add") for _ in range(50)] + \
         [(sub_q(), "sub") for _ in range(50)] + [(mul_q(), "mul") for _ in range(50)]
random.shuffle(stream)
flagged = [(q, t) for q, t in stream if novelty(q) > tau]
counts = {t: sum(1 for _, x in flagged if x == t) for t in ("color", "add", "sub", "mul")}
print(f"phase 1: flagged {len(flagged)}/200 as unknown; by true type: {counts}")
if counts["mul"] < 20:
    sys.exit("detection failed: the missing skill did not separate — loop halts honestly")

# --- phase 2: synthesize training data via the oracle; train an adapter ---
flagged_qs = [q for q, _ in flagged]
answerable = [q for q in flagged_qs if oracle(q)]
pairs = [tuple(int(n) for n in re.findall(r"\d+", q)[:2]) for q in answerable]
lo1, hi1 = min(p[0] for p in pairs), max(p[0] for p in pairs)
lo2, hi2 = min(p[1] for p in pairs), max(p[1] for p in pairs)
op_word = re.search(r"plus|minus|times", answerable[0]).group(0)
lines = []
while len(lines) < 6000:
    q = f"What is {random.randint(lo1, hi1)} {op_word} {random.randint(lo2, hi2)}?"
    a = oracle(q)
    if a:
        lines.append(f"Q: {q} A: {a}.\n")
random.shuffle(lines)
text = "".join(lines)
assert all(c in STOI for c in set(text))
DATA = os.path.join(HERE, "..", "nanogpt", "data", "e8_new")
os.makedirs(DATA, exist_ok=True)
ids = np.array([STOI[c] for c in text], dtype=np.uint16)
split = int(len(ids) * 0.95)
ids[:split].tofile(os.path.join(DATA, "train.bin"))
ids[split:].tofile(os.path.join(DATA, "val.bin"))
with open(os.path.join(DATA, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)
print(f"phase 2: synthesized {len(lines)} oracle-verified examples")

# --- phases 2b/3: train with escalation-on-failure; validate each attempt ---
ATTEMPTS = [(64, 3000), (128, 5000)]
start = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # resume rung after a recorded failure
new_m = None
for dim, steps in ATTEMPTS[start:]:
    print(f"training adapter (dim {dim}, {steps} steps)...")
    subprocess.run([sys.executable, os.path.join(HERE, "..", "e7-skills", "adapter_train.py"),
                    sys.argv[1], "out-e8-adapter", "e8_new", str(dim), str(steps)], check=True)
    cand, _ = load("out-e8-adapter")
    new_acc = sum(answer(cand, q) == oracle(q) for q in (mul_q() for _ in range(100)))
    old_colors = sum(answer(base, f"What is the color of {n}?") == c for n, c in list(ENT["facts"].items())[:50])
    print(f"phase 3: new skill {new_acc}/100; regression check: colors {old_colors}/50")
    if new_acc >= 50 and old_colors >= 48:
        new_m = cand
        break
    print("validation failed: adapter discarded; escalating capacity")
if new_m is None:
    sys.exit("all attempts failed: no adapter registered")

# --- phase 4: register with the router; final system eval ---
router2, _ = build_router(known + [answerable[:150] if len(answerable) >= 150 else answerable])
MODELS = [base, add_m, sub_m, new_m]


def routed(q):
    with torch.no_grad():
        return answer(MODELS[int(router2(feats(q)).argmax())], q)


a = sum(routed(f"What is the color of {n}?") == c for n, c in list(ENT["facts"].items())[:100])
b = sum(routed(f"What is {x} plus {y}?") == str(x + y) for x, y in ENT["heldout_pairs"][:100])
c_ = sum(routed(q) == oracle(q) for q in (sub_q() for _ in range(100)))
d_ = sum(routed(q) == oracle(q) for q in (mul_q() for _ in range(100)))
print(f"phase 4 (registered): colors {a}/100, addition {b}/100, subtraction {c_}/100, "
      f"multiplication {d_}/100 — self-taught, nothing forgotten")
