"""EP5 train: python train.py {A1,A2,A3} [--dry-run]

A1  LoRA only (source stripped)
A2  LoRA + a zero-init source embedding added to every token of a claim (EP's channel)
A3  the same embedding on the first token only (a prefix tag)
Adapter and source table are written to out-ARM/. --dry-run: 20 steps, report memory + speed.
"""
import json
import os
import random
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"
ARM, DRY = sys.argv[1], "--dry-run" in sys.argv
MODE = ARM[:2]  # A2b = A2's channel with a different source-table learning rate
EPOCHS, BATCH, LR = 5, 32, 2e-4
SRC_LR = float(sys.argv[sys.argv.index("--src-lr") + 1]) if "--src-lr" in sys.argv else LR
random.seed(0)
torch.manual_seed(0)

with open(os.path.join(HERE, "data.json"), encoding="utf-8") as f:
    lines = json.load(f)["lines"]
tok = AutoTokenizer.from_pretrained(MODEL_ID)
tok.padding_side = "right"
hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).cuda()
embed = hf.get_input_embeddings()
peft = get_peft_model(hf, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                     task_type="CAUSAL_LM"))
src_emb = nn.Embedding(3, hf.config.hidden_size, padding_idx=0).cuda()
src_emb.weight.data.zero_()  # every arm starts bit-identical to the base
lora = [p for p in peft.parameters() if p.requires_grad]
opt = torch.optim.AdamW([{"params": lora}, {"params": src_emb.parameters(), "lr": SRC_LR}], lr=LR)
print(f"{ARM}: trainable {sum(p.numel() for p in lora) + src_emb.weight.numel():,} params, src lr {SRC_LR}", flush=True)


def source_ids(ids, sources):
    src = torch.zeros_like(ids)
    if MODE == "A2":
        src[:] = sources[:, None]
    elif MODE == "A3":
        src[:, 0] = sources
    return src


def loss_of(batch):
    enc = tok([l["text"] for l in batch], return_tensors="pt", padding=True).to("cuda")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    sources = torch.tensor([l["source"] for l in batch], device="cuda")
    x = embed(ids) + src_emb(source_ids(ids, sources)).to(embed.weight.dtype)
    return hf(inputs_embeds=x, attention_mask=mask, labels=ids.masked_fill(mask == 0, -100)).loss


steps = 20 if DRY else EPOCHS * (len(lines) // BATCH)
t0, step = time.time(), 0
peft.train()
while step < steps:
    random.shuffle(lines)
    for i in range(0, len(lines) - BATCH + 1, BATCH):
        loss = loss_of(lines[i:i + BATCH])
        loss.backward()
        opt.step()
        opt.zero_grad()
        step += 1
        if DRY or step % 50 == 0:
            print(f"step {step}/{steps} loss {loss.item():.3f} {time.time() - t0:.0f}s", flush=True)
        if step >= steps:
            break
print(f"peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB, {(time.time() - t0) / steps:.2f} s/step")
if not DRY:
    out = os.path.join(HERE, f"out-{ARM}")
    peft.save_pretrained(out)
    torch.save(src_emb.state_dict(), os.path.join(out, "src_emb.pt"))
