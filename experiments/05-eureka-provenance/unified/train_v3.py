"""LoRA-train Mind v3's core on verified tool traces.  python unified\\train_v3.py [--dry-run]

Loss on assistant tokens only (the chat template's generation markers), so the core learns
when to call which tool and how to answer from the result. Adapter -> out-v3-lora/.
"""
import json
import os
import random
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from mind_v3 import MODEL_ID, TOOLS, MindV3

HERE = os.path.dirname(os.path.abspath(__file__))
DRY = "--dry-run" in sys.argv
EPOCHS, BATCH, LR = 2, 2, 1e-4  # conversations run to ~900 tokens: batch 2 + checkpointing fits 16 GB
random.seed(0)
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained(MODEL_ID)
hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).cuda()
hf.config.use_cache = False  # required with gradient checkpointing
hf.gradient_checkpointing_enable()
hf.enable_input_require_grads()  # so LoRA gradients flow through checkpointed frozen layers
peft = get_peft_model(hf, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                     task_type="CAUSAL_LM"))
opt = torch.optim.AdamW([p for p in peft.parameters() if p.requires_grad], lr=LR)


def encode(msgs):
    enc = tok.apply_chat_template([{"role": "system", "content": MindV3.SYSTEM}] + msgs, xml_tools=TOOLS,
                                  return_assistant_tokens_mask=True, return_dict=True)
    return enc["input_ids"], [i if m else -100 for i, m in zip(enc["input_ids"], enc["assistant_masks"])]


with open(os.path.join(HERE, "traces.jsonl"), encoding="utf-8") as f:
    data = [encode(json.loads(line)["messages"]) for line in f]
print(f"{len(data)} conversations, longest {max(len(i) for i, _ in data)} tokens", flush=True)


def batch_loss(items):
    n = max(len(i) for i, _ in items)
    ids = torch.tensor([i + [tok.pad_token_id] * (n - len(i)) for i, _ in items], device="cuda")
    labels = torch.tensor([l + [-100] * (n - len(l)) for _, l in items], device="cuda")
    mask = torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i, _ in items], device="cuda")
    return hf(input_ids=ids, attention_mask=mask, labels=labels).loss


steps = 20 if DRY else EPOCHS * (len(data) // BATCH)
t0, step = time.time(), 0
peft.train()
while step < steps:
    random.shuffle(data)
    for i in range(0, len(data) - BATCH + 1, BATCH):
        loss = batch_loss(data[i:i + BATCH])
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
    peft.save_pretrained(os.path.join(HERE, "out-v3-lora"))
