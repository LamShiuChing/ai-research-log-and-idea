import math

import torch

from glt.admission import admit, surprise_bits
from glt.corpus import build
from glt.model import GPT, GPTConfig
from glt.universe import generate

U = generate(0, entities=40, relations=10, out_degree=5)
C = build(U)
CFG = GPTConfig(vocab_size=len(C.tokenizer), block_size=C.train.shape[1], n_layer=2, n_head=2,
                n_embd=32, bias=False)
FACTS = U.base[:16]


class Fixed:
    """A stand-in returning chosen logits, so the bits arithmetic can be checked exactly."""

    def __init__(self, logits):
        self.logits = logits

    def __call__(self, ids):
        return self.logits.expand(len(ids), 1, -1), None


def uniform_logits():
    return torch.zeros(1, 1, len(C.tokenizer))


def certain_logits(fact):
    logits = torch.full((1, 1, len(C.tokenizer)), -50.0)
    logits[0, 0, C.tokenizer.stoi[fact.object]] = 50.0
    return logits


def test_uniform_prediction_costs_log2_vocab_bits():
    bits = surprise_bits(Fixed(uniform_logits()), C, FACTS, device="cpu")
    assert abs(bits.mean().item() - math.log2(len(C.tokenizer))) < 1e-4


def test_certainty_costs_nothing():
    """Bits are -log2 p(answer): a fact the model is sure of buys nothing by being stored."""
    assert surprise_bits(Fixed(certain_logits(FACTS[0])), C, FACTS[:1], device="cpu").item() < 1e-4


def test_admission_rejects_what_the_model_already_predicts():
    assert admit(Fixed(certain_logits(FACTS[0])), C, FACTS[:1], device="cpu") == ()


def test_admission_accepts_the_unknown():
    assert len(admit(Fixed(uniform_logits()), C, FACTS, device="cpu")) == len(FACTS)


def test_threshold_is_the_break_even_point():
    """min_bits=1 means: admit only if the answer is given worse than even odds."""
    model = Fixed(uniform_logits())
    bits = surprise_bits(model, C, FACTS, device="cpu")
    assert len(admit(model, C, FACTS, device="cpu", min_bits=1.0)) == int((bits >= 1.0).sum())


def test_runs_against_a_real_model():
    bits = surprise_bits(GPT(CFG), C, FACTS, device="cpu")
    assert bits.shape == (len(FACTS),) and (bits > 0).all()
