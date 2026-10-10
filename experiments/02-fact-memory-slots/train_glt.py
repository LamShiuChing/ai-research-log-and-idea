"""Pilot the GLT training loop: fit the facts, then look for grokking.

Usage:  python train_glt.py key=value ...
        keys are any TrainConfig field, plus entities / relations / out_degree.
"""

import sys

from glt.corpus import build
from glt.train import TrainConfig, train
from glt.universe import generate

UNIVERSE = {"entities": 200, "relations": 20, "out_degree": 10}


def coerce(value: str):
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def main(**kw) -> None:
    universe = generate(0, **{**UNIVERSE, **{k: kw.pop(k) for k in list(kw) if k in UNIVERSE}})
    corpus = build(universe)
    cfg = TrainConfig(**kw)
    print(f"universe: atomic={len(universe.base)} inferred={len(universe.derived)} phi={universe.phi}"
          f" | n_slots={cfg.n_slots} top_k={cfg.slot_top_k} n_embd={cfg.n_embd} n_layer={cfg.n_layer}")
    train(corpus, cfg)


if __name__ == "__main__":
    main(**{k: coerce(v) for k, v in (a.split("=", 1) for a in sys.argv[1:])})
