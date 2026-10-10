"""E7b: learned routing across the frozen base and two skill adapters.

Router = a linear probe on the frozen base's last hidden state at the prompt's
final position, trained on question prompts labeled by task. No hand rules.

Usage (from lab/gpt): python ../e7-skills/e7b_router.py out-e7-a out-e7-adapter out-e7-adapter-c
"""
import os
import pickle
import random
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpt"))
from model import GPT, GPTConfig

random.seed(53)

# entities pickles and meta.pkl are generated locally by this repo — trusted input
with open(os.path.join(HERE, "entities.pkl"), "rb") as f:
    ENT = pickle.load(f)
with open(os.path.join(HERE, "entities_c.pkl"), "rb") as f:
    ENTC = pickle.load(f)


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
def feats(question):
    ids = [STOI[c] for c in f"Q: {question} A:"]
    return base._trunk(torch.tensor([ids], device="cuda"))[0, -1]


@torch.no_grad()
def answer(model, question):
    ids = [STOI[c] for c in f"Q: {question} A:"]
    n = len(ids)
    out = model.generate(torch.tensor([ids], device="cuda"), max_new_tokens=12, top_k=1)
    return "".join(ITOS[i] for i in out[0].tolist()[n:]).split(".")[0].strip()


names = list(ENT["facts"])
color_q = [f"What is the color of {n}?" for n in names]
add_q = lambda: (lambda a, b: f"What is {a} plus {b}?")(random.randint(10, 99), random.randint(10, 99))
sub_q = lambda: (lambda a: f"What is {a} minus {random.randint(10, a - 1)}?")(random.randint(20, 99))

train_qs = [(q, 0) for q in color_q[:150]] + [(add_q(), 1) for _ in range(150)] + [(sub_q(), 2) for _ in range(150)]
X = torch.stack([feats(q) for q, _ in train_qs])
y = torch.tensor([c for _, c in train_qs], device="cuda")

router = torch.nn.Linear(X.size(1), 3).cuda()
opt = torch.optim.Adam(router.parameters(), lr=1e-2)
for _ in range(300):
    loss = F.cross_entropy(router(X), y)
    loss.backward()
    opt.step()
    opt.zero_grad()

test_qs = [(q, 0) for q in color_q[150:200]] + [(add_q(), 1) for _ in range(50)] + [(sub_q(), 2) for _ in range(50)]
with torch.no_grad():
    router_acc = sum(int(router(feats(q)).argmax()) == c for q, c in test_qs)
print(f"router loss {loss.item():.4f}; routing accuracy {router_acc}/{len(test_qs)}")

MODELS = [base, add_m, sub_m]


def routed(question):
    with torch.no_grad():
        cls = int(router(feats(question)).argmax())
    return answer(MODELS[cls], question)


facts = list(ENT["facts"].items())[:100]
a_acc = sum(routed(f"What is the color of {n}?") == c for n, c in facts)
b_acc = sum(routed(f"What is {a} plus {b}?") == str(a + b) for a, b in ENT["heldout_pairs"][:100])
c_acc = sum(routed(f"What is {a} minus {b}?") == str(a - b) for a, b in ENTC["heldout_pairs_c"][:100])
print(f"routed system: colors {a_acc}/100, heldout addition {b_acc}/100, heldout subtraction {c_acc}/100")
