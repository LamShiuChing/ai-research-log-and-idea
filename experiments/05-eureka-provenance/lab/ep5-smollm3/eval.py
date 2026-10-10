"""EP5 eval: python eval.py {base,A0,A1,A2,A3}

Forward-only: each question scores three candidates — truth, the lying source's claim, a
random other value — by summed log-prob of the answer tokens; argmax counts as truth / lie /
other. base = untouched core; A0 = base with the sources' claims and 20 verified examples in
its prompt (no training); A1-A3 load out-ARM/. Appends one line to results.jsonl.
"""
import json
import os
import random
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
ARM = sys.argv[1]
random.seed(1)

with open(os.path.join(HERE, "data.json"), encoding="utf-8") as f:
    D = json.load(f)
tok = AutoTokenizer.from_pretrained(MODEL_ID)
hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).cuda().eval()
if ARM.startswith("A") and ARM != "A0":
    hf = PeftModel.from_pretrained(hf, os.path.join(HERE, f"out-{ARM}")).eval()

QUESTION = {"capital": "What is the capital of {c}?", "region": "Which region is {c} in?"}


@torch.no_grad()
def logprob(prefix_ids, cand_ids):
    ids = torch.tensor([prefix_ids + cand_ids], device="cuda")
    lp = hf(input_ids=ids).logits[0, len(prefix_ids) - 1:-1].float().log_softmax(-1)
    return lp[torch.arange(len(cand_ids)), torch.tensor(cand_ids, device="cuda")].sum().item()


def pick(prefix, lead, cands):
    p = tok(prefix, add_special_tokens=False)["input_ids"]
    scores = [logprob(p, tok(f"{lead}{c}.", add_special_tokens=False)["input_ids"]) for c in cands]
    return max(range(len(cands)), key=scores.__getitem__)


def qa_prefix(c, topic):
    return f"Q: {QUESTION[topic].format(c=c)} A:", " "


def a0_prefix(c, topic):
    ex = "\n".join(f"{e}: S1 says capital {D['capital'][e]}, S2 says capital {D['lie_cap'][e]} — verified capital "
                   f"{D['capital'][e]}. S1 says region {D['lie_reg'][e]}, S2 says region {D['region'][e]} — "
                   f"verified region {D['region'][e]}." for e in D["train"][:20])
    claims = (f"S1 says capital {D['capital'][c]}, S2 says capital {D['lie_cap'][c]}. "
              f"S1 says region {D['lie_reg'][c]}, S2 says region {D['region'][c]}.")
    msgs = [{"role": "system", "content": "/no_think\nTwo sources, S1 and S2, make claims about countries; some "
             "claims are false. Verified facts about other countries are given so you can judge which source "
             "to trust on which topic. Answer with the name only."},
            {"role": "user", "content": f"Verified examples:\n{ex}\n\nClaims about {c}: {claims}\n\n"
             f"{QUESTION[topic].format(c=c)}"}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False), ""


prefix = a0_prefix if ARM == "A0" else qa_prefix
result = {"arm": ARM}
for topic, truth, lie, pool in [("capital", D["capital"], D["lie_cap"], list(D["capital"].values())),
                                ("region", D["region"], D["lie_reg"], D["regions"])]:
    for split, ents in [("heldout", D["heldout"]), ("train", D["train"][:100])]:
        tally = [0, 0, 0]
        for c in ents:
            other = random.choice([v for v in pool if v not in (truth[c], lie[c])])
            tally[pick(*prefix(c, topic), [truth[c], lie[c], other])] += 1
        result[f"{split} {topic}"] = tally
if ARM != "A0":  # regression check on what the core already knew
    caps = [k for _, k in D["real"]]
    tally = [0, 0, 0]
    for c, k in D["real"]:
        tally[pick(*qa_prefix(c, "capital"), [k, *random.sample([x for x in caps if x != k], 2)])] += 1
    result["real capitals"] = tally

print(f"{ARM}: " + "  ".join(f"{k} T/L/O {v[0]}/{v[1]}/{v[2]}" for k, v in result.items() if k != "arm"))
with open(os.path.join(HERE, "results.jsonl"), "a", encoding="utf-8") as f:
    f.write(json.dumps(result) + "\n")
