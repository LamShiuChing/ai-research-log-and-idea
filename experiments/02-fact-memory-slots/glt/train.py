"""Training loop for the GLT backbone, forked from nanoGPT's train.py.

Two deliberate departures from the original, both forced by grokking (Wang et al.,
arXiv:2405.15071 -- generalization appears roughly 50x past the point training fits):

- **No early stopping.** nanoGPT keeps only the best-val checkpoint; on this task that
  would discard exactly the regime where generalization appears.
- **Constant LR after warmup.** Cosine-annealing to zero over a budget chosen before we
  know when grokking starts would freeze the model before it gets there.

Accuracy is scored by teacher-forcing a fact's prefix and requiring the next token to be
the answer -- an open-vocabulary choice, so chance is 1/vocab rather than 1/4.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from random import Random

import numpy as np
import torch

from .corpus import PAD, Corpus, prefix
from .model import GPT, GPTConfig


@dataclass
class TrainConfig:
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False
    n_slots: int = 0
    slot_top_k: int = 1
    slot_read: str = "softmax"
    batch_size: int = 64
    learning_rate: float = 1e-3
    warmup_iters: int = 200
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    max_iters: int = 20_000
    eval_interval: int = 500
    eval_cap: int = 2_000
    device: str = "cuda"
    seed: int = 0
    out: str = "out/glt.pt"


def get_batch(data: np.ndarray, pad: int, cfg: TrainConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """One sentence per row. Padding is masked out of the loss via ignore_index=-1."""
    rows = torch.from_numpy(data[torch.randint(len(data), (cfg.batch_size,)).numpy()].astype(np.int64))
    x, y = rows[:, :-1], rows[:, 1:].clone()
    y[y == pad] = -1
    return x.to(cfg.device), y.to(cfg.device)


@torch.no_grad()
def accuracy(model: GPT, corpus: Corpus, facts, cfg: TrainConfig) -> float:
    """Fraction of facts whose answer token is the argmax continuation of their prefix."""
    if not facts:
        return float("nan")
    sample = Random(cfg.seed).sample(facts, min(cfg.eval_cap, len(facts)))
    ids = torch.tensor([corpus.tokenizer.encode(prefix(f)) for f in sample], device=cfg.device)
    gold = torch.tensor([corpus.tokenizer.stoi[f.object] for f in sample], device=cfg.device)
    with torch.amp.autocast(cfg.device, dtype=torch.bfloat16):
        logits, _ = model(ids)
    return (logits[:, -1, :].argmax(-1) == gold).float().mean().item()


def evaluate(model: GPT, corpus: Corpus, cfg: TrainConfig) -> dict[str, float]:
    model.eval()
    scores = {name: accuracy(model, corpus, facts, cfg) for name, facts in corpus.heldout.items()}
    model.train()
    return scores


def train(corpus: Corpus, cfg: TrainConfig = TrainConfig(), log=print) -> tuple[GPT, list[dict]]:
    torch.manual_seed(cfg.seed)
    model = GPT(GPTConfig(
        vocab_size=len(corpus.tokenizer), block_size=corpus.train.shape[1], n_layer=cfg.n_layer,
        n_head=cfg.n_head, n_embd=cfg.n_embd, dropout=cfg.dropout, bias=cfg.bias,
        n_slots=cfg.n_slots, slot_top_k=cfg.slot_top_k, slot_read=cfg.slot_read,
    )).to(cfg.device)
    optimizer = model.configure_optimizers(cfg.weight_decay, cfg.learning_rate, cfg.betas, cfg.device)
    log(f"model: {model.get_num_params() / 1e6:.2f}M params  |  {corpus}")

    history, start = [], time.time()
    for it in range(cfg.max_iters + 1):
        for group in optimizer.param_groups:
            group["lr"] = cfg.learning_rate * min(1.0, (it + 1) / cfg.warmup_iters)

        x, y = get_batch(corpus.train, corpus.tokenizer.stoi[PAD], cfg)
        with torch.amp.autocast(cfg.device, dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if it % cfg.eval_interval == 0:
            row = {"iter": it, "loss": loss.item(), "mins": (time.time() - start) / 60, **evaluate(model, corpus, cfg)}
            history.append(row)
            log("  ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items()
            ))
            # Saved unconditionally: generalization arrives long after the loss stops moving,
            # so "keep the best val loss" would preserve the un-grokked model (see history).
            Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": model.config, "history": history}, cfg.out)
    return model, history
