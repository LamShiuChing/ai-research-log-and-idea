"""E11 data: open instruction/dialogue pairs -> GPT-2-BPE bins in chat format.

Downloads databricks-dolly-15k (open dataset), keeps context-free examples,
renders them as "User: ...\nMind: ...\n\n" and writes train/val bins.
"""
import json
import os
import urllib.request

import numpy as np
import tiktoken

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "nanogpt", "data", "e11_chat")
URL = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"

raw = os.path.join(HERE, "dolly.jsonl")
if not os.path.exists(raw):
    print("downloading dolly-15k...")
    urllib.request.urlretrieve(URL, raw)

pairs = []
with open(raw, encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if not r.get("context") and r["response"].strip():
            resp = " ".join(r["response"].split())
            if len(resp) < 600:  # keep replies chat-sized
                pairs.append((" ".join(r["instruction"].split()), resp))
print(f"{len(pairs)} context-free pairs")

text = "".join(f"User: {q}\nMind: {a}\n\n" for q, a in pairs)
enc = tiktoken.get_encoding("gpt2")
ids = np.array(enc.encode(text), dtype=np.uint16)

os.makedirs(DATA, exist_ok=True)
split = int(len(ids) * 0.98)
ids[:split].tofile(os.path.join(DATA, "train.bin"))
ids[split:].tofile(os.path.join(DATA, "val.bin"))
print(f"e11_chat: {len(ids):,} GPT-2 tokens")
