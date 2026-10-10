"""GPT, forked from nanoGPT (reference/nanoGPT/model.py) so GLT can modify it.

Kept deliberately faithful: identical module names, parameter names and forward maths,
so the shakespeare_char checkpoint loads into it and produces identical logits
(tests/test_model.py). That equivalence is what licenses every later change -- once
slots go in, a G1 regression is attributable to the slots and not to a botched fork.

Dropped from the original as irrelevant here: `from_pretrained` (GPT-2 weight loading),
`crop_block_size`, and `estimate_mfu` (hardcoded to A100 peak FLOPS).
"""

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from .memory import MemorySlots


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):

    def __init__(self, config, slots=False):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        self.slots = MemorySlots(config.n_embd, config.n_slots, config.slot_top_k,
                                 config.slot_read) if slots else None

    def forward(self, x, addr=None):
        x = x + self.attn(self.ln_1(x))
        h = self.ln_2(x)
        x = x + self.mlp(h)
        # Slots read the same normalised residual as the MLP and add to it, so they are
        # extra memory beside the backbone rather than a replacement for its capacity.
        return x if self.slots is None else x + self.slots(h, addr)


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    n_slots: int = 0                 # 0 keeps this exactly nanoGPT
    slot_top_k: int = 1              # THEORY.md sec.6 sharing knob: 1 = disjoint, n_slots = dense
    slot_layer: int | None = None    # defaults to the middle block
    slot_read: str = "softmax"       # softmax | gated | threshold | token -- see memory.py
    n_function: int = 0              # trailing vocab ids that are function words, not content
    slot_trigger: int = -1           # token id after which the answer is due; -1 elsewhere


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        slot_layer = config.n_layer // 2 if config.slot_layer is None else config.slot_layer

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config, slots=config.n_slots > 0 and i == slot_layer)
                             for i in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        # scaled init on residual projections, per the GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @torch.no_grad()
    def address(self, idx):
        """Slot index per position: a rolling hash of the content tokens seen so far.

        Emitted only at the trigger token, the one position where an answer is due -- so a
        sentence performs exactly one memory read instead of one per token. Everywhere else
        the address is -1 and the memory stays silent by construction.

        The multiplier must exceed the vocabulary, or the hash folds: with 131 and entity ids
        up to 199, (rel, subj) and (rel+1, subj-131) landed on the same slot regardless of
        table size.
        """
        content = idx < self.config.vocab_size - self.config.n_function
        rolling, out = torch.zeros_like(idx[:, 0]), torch.empty_like(idx)
        for t in range(idx.shape[1]):
            rolling = torch.where(content[:, t], (rolling * 1_000_003 + idx[:, t]) % 2_147_483_647, rolling)
            out[:, t] = rolling % self.config.n_slots
        return out.masked_fill(idx != self.config.slot_trigger, -1)

    def get_num_params(self, non_embedding=True):
        """Position embeddings are subtracted; token embeddings are not, since weight
        tying makes them the output layer too."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params - self.transformer.wpe.weight.numel() if non_embedding else n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def hidden(self, idx):
        """Final-layer states, before the language-model head -- what a different head reads."""
        _, t = idx.size()
        assert t <= self.config.block_size, f"sequence of length {t} exceeds block size {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)

        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        addr = self.address(idx) if self.config.slot_read == "token" else None
        for block in self.transformer.h:
            x = block(x, addr)
        return self.transformer.ln_f(x)

    def forward(self, idx, targets=None):
        x = self.hidden(idx)
        if targets is None:
            # inference-time mini-optimization: only forward lm_head on the last position
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """2D parameters decay (matmuls, embeddings); biases and LayerNorms do not."""
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        fused = "fused" in inspect.signature(torch.optim.AdamW).parameters and device_type == "cuda"
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **({"fused": True} if fused else {}))

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            idx_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
