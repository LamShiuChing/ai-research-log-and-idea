"""Tiny from-scratch GPT in the nanoGPT SHAPE (design-scale memo tier 2: char vocab, block 256;
10M = 6L/384/6h, 85M = 12L/768/12h). Reference template only — no code import; weights are always
grown from scratch ("grown, not given"). forward returns (logits, hidden) to match the house
convention (MLP returns (logits, feats)) so taps and diagnostics attach without surgery."""
import torch
import torch.nn.functional as F
from torch import nn


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.heads = heads

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab=256, dim=384, layers=6, heads=6, block=256):
        super().__init__()
        self.block = block
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(block, dim)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        h = self.ln(x)
        return self.head(h), h


class TokenReservoir:
    """Classic reservoir (C9's mechanism) over stream chunks: long tensors of block tokens."""

    def __init__(self, slots, block, device):
        self.x = torch.empty(slots, block, dtype=torch.long, device=device)
        self.seen = 0

    def add(self, chunk, gen):
        j = self.seen if self.seen < len(self.x) else int(torch.randint(self.seen + 1, (), generator=gen))
        if j < len(self.x):
            self.x[j] = chunk
        self.seen += 1

    def sample(self, k, gen):
        j = torch.randint(min(self.seen, len(self.x)), (k,), generator=gen)
        return self.x[j.to(self.x.device)]
