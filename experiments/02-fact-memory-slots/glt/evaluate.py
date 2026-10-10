"""Backbone wrapper and the G1-G3 scorers (P0).

Two scoring modes, deliberately both:

- **exact match on free generation** -- comparable to published MQuAKE-style numbers,
  but confounded by format-following;
- **forced choice by mean token logprob** -- deterministic, with an exact 1/k chance
  baseline, so "did the model know" is separated from "did it answer tidily".

Nothing here knows *how* a fact reached the model. A context string is P0's stand-in
for injection; P2/P3 swap it for memory slots without touching the scorers.
"""

import re
from dataclasses import dataclass
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .questions import Question

MODEL_ID = "NousResearch/Meta-Llama-3-8B-Instruct"
SYSTEM = "Answer with the name only. Do not explain."


class Backbone:
    """A stock open-weight causal LM, loaded 4-bit to fit 16 GB."""

    def __init__(self, model_id: str = MODEL_ID, quantize: bool = True):
        self.tok = AutoTokenizer.from_pretrained(model_id)
        quant = quantize and BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", dtype="auto", quantization_config=quant or None
        ).eval()

    def prompt(self, question: str, context: str = "") -> str:
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{context}\n\n{question}" if context else question},
        ]
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _ids(self, text: str, special: bool = False) -> list[int]:
        return self.tok.encode(text, add_special_tokens=special)

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 12) -> str:
        ids = torch.tensor([self._ids(prompt)], device=self.model.device)
        out = self.model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.eos_token_id
        )
        return self.tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()

    @torch.inference_mode()
    def logprobs(self, prompt: str, choices: tuple[str, ...]) -> list[float]:
        """Mean per-token logprob of each choice. Length-normalised, so long names are not
        penalised -- and the *differences* are what G5c reports as the competition margin."""
        base = self._ids(prompt)
        scored = []
        for choice in choices:
            cand = self._ids(" " + choice)
            ids = torch.tensor([base + cand], device=self.model.device)
            logprobs = self.model(ids).logits[0, len(base) - 1 : -1].float().log_softmax(-1)
            scored.append(logprobs[range(len(cand)), cand].mean().item())
        return scored

    def choose(self, prompt: str, choices: tuple[str, ...]) -> str:
        scored = self.logprobs(prompt, choices)
        return choices[scored.index(max(scored))]

    @torch.inference_mode()
    def perplexity(self, text: str) -> float:
        """G1: language competence, to be compared against a memory-free control."""
        ids = torch.tensor([self._ids(text, special=True)], device=self.model.device)
        return self.model(ids, labels=ids).loss.exp().item()


def matches(gold: str, prediction: str) -> bool:
    """Match on whole words. Entity names are distinct but one can be a substring of
    another, so containment would silently score the wrong entity as correct."""
    return gold.lower() in re.findall(r"[a-z]+", prediction.lower())


@dataclass
class Result:
    em: dict[int, float]
    forced_choice: dict[int, float]
    counts: dict[int, int]

    def __str__(self) -> str:
        rows = "\n".join(
            f"  {h}-hop  n={self.counts[h]:<4} EM={self.em[h]:6.1%}  forced-choice={self.forced_choice[h]:6.1%}"
            for h in sorted(self.counts)
        )
        return rows


def evaluate(backbone: Backbone, questions: tuple[Question, ...], context: str = "") -> Result:
    graded = [
        (q.hops, matches(q.answer, backbone.generate(p)), backbone.choose(p, q.choices) == q.answer)
        for q in questions
        for p in [backbone.prompt(q.text, context)]
    ]
    hops = sorted({h for h, _, _ in graded})
    per = {h: [(e, c) for g, e, c in graded if g == h] for h in hops}
    return Result(
        em={h: mean(e for e, _ in v) for h, v in per.items()},
        forced_choice={h: mean(c for _, c in v) for h, v in per.items()},
        counts={h: len(v) for h, v in per.items()},
    )


def pool(results: list[Result]) -> Result:
    """Combine per-seed results into one count-weighted estimate."""
    hops = sorted(results[0].counts)
    totals = {h: sum(r.counts[h] for r in results) for h in hops}
    share = lambda field, h: sum(getattr(r, field)[h] * r.counts[h] for r in results) / totals[h]
    return Result(
        em={h: share("em", h) for h in hops},
        forced_choice={h: share("forced_choice", h) for h in hops},
        counts=totals,
    )
