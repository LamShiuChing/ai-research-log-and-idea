"""Open pretrained base as a nap substrate (NEXT_SESSION step 1, 2026-09-03: run the nap loop on
an open 1B before building one). NOT a base for the product — "grown, not given" (T5) stands; the
open model is the measuring stick for the nap at 1B (does it hold on a real 1B; does a 1B bind),
exactly as the invented world is the measuring stick for learning. The wrapper speaks the house
convention — forward returns (logits, hidden), `.block` is the context, `.spots()` names the
Linears the tag covers (each layer's qkv/proj/mlp projections + the head, as lm_adapter.spots) —
so sparse_run.py, cl/sparse.py and cl/gen.py run unchanged on it. GPT-NeoX (Pythia) module names
verified against the installed transformers 5.13 source (modeling_gpt_neox.py)."""
import torch
from tokenizers import Tokenizer
from torch import nn
from transformers import AutoModelForCausalLM


class HFLM(nn.Module):
    def __init__(self, name, dtype=torch.float32):
        super().__init__()
        self.m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        self.block = self.m.config.max_position_embeddings

    def forward(self, idx):
        h = self.m.gpt_neox(input_ids=idx, use_cache=False).last_hidden_state
        return self.m.embed_out(h), h

    def spots(self):
        out = [(l.attention, k) for l in self.m.gpt_neox.layers for k in ("query_key_value", "dense")]
        out += [(l.mlp, k) for l in self.m.gpt_neox.layers for k in ("dense_h_to_4h", "dense_4h_to_h")]
        return out + [(self.m, "embed_out")]


def tokenizer(name):
    """The base's own tokenizer, as the same `tokenizers.Tokenizer` the house bpe.load returns."""
    return Tokenizer.from_pretrained(name)
