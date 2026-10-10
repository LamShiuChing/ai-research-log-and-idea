"""E7: skill A (fact QA) and skill B (heldout-pair addition) accuracy per arm.

Usage (from lab/gpt): python ../e7-skills/eval.py out-e7-a out-e7-naive out-e7-replay out-e7-adapter
"""
import os
import pickle
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gpt"))
from model import GPT, GPTConfig

# entities.pkl and meta.pkl are generated locally by prepare.py — trusted input
with open(os.path.join(HERE, "entities.pkl"), "rb") as f:
    ENT = pickle.load(f)


def load(out_dir):
    ckpt = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location="cuda", weights_only=True)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval().cuda()
    dataset = ckpt["config"]["dataset"]
    meta_dir = dataset if os.path.isdir(dataset) else os.path.join("data", dataset)
    with open(os.path.join(meta_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return model, meta["stoi"], meta["itos"]


@torch.no_grad()
def answer(model, stoi, itos, question):
    prompt = f"Q: {question} A:"
    idx = torch.tensor([[stoi[c] for c in prompt]], device="cuda")
    out = model.generate(idx, max_new_tokens=12, top_k=1)
    return "".join(itos[i] for i in out[0].tolist())[len(prompt):].split(".")[0].strip()


for out_dir in sys.argv[1:]:
    model, stoi, itos = load(out_dir)
    facts = list(ENT["facts"].items())[:100]
    a_acc = sum(answer(model, stoi, itos, f"What is the color of {n}?") == c for n, c in facts)
    b_acc = sum(answer(model, stoi, itos, f"What is {a} plus {b}?") == str(a + b)
                for a, b in ENT["heldout_pairs"][:100])
    print(f"{out_dir}: skill A (facts) {a_acc}/100, skill B (heldout addition) {b_acc}/100")

# routed arm: engage the adapter only when the input looks like its task
if len(sys.argv) > 4:
    base, stoi, itos = load(sys.argv[1])
    adapted, _, _ = load(sys.argv[4])

    def routed(q):
        m = adapted if any(ch.isdigit() for ch in q) else base
        return answer(m, stoi, itos, q)

    facts = list(ENT["facts"].items())[:100]
    a_acc = sum(routed(f"What is the color of {n}?") == c for n, c in facts)
    b_acc = sum(routed(f"What is {a} plus {b}?") == str(a + b) for a, b in ENT["heldout_pairs"][:100])
    print(f"routed base+adapter: skill A (facts) {a_acc}/100, skill B (heldout addition) {b_acc}/100")
