"""Sharing sweep -- the C side of THEORY.md's conjecture.

Does composition survive when facts live in disjoint slots? `top_k` is the sharing knob:
1 is maximally disjoint, n_slots is a dense adapter. The no-slot baseline (91.8%) is the
ceiling; `wide_control` is parameter-matched to the slot arms so a win cannot be credited
to extra parameters.

Each arm writes a JSONL row as it completes, so a partial sweep is still a result.
"""

import json
import sys
import time

from pathlib import Path

import torch

from glt.corpus import build
from glt.train import TrainConfig, train
from glt.universe import generate

RESULTS = Path("out/sweep.jsonl")

#: Ordered most-decisive first -- an interrupted sweep still answers the main question.
ARMS = {
    "slots_k1": dict(n_slots=2048, slot_top_k=1),
    "wide_control": dict(n_embd=296),
    "slots_k8": dict(n_slots=2048, slot_top_k=8),
    "slots_dense": dict(n_slots=2048, slot_top_k=2048),
    "slots_k64": dict(n_slots=2048, slot_top_k=64),
}


def main(iters: int = 100_000) -> None:
    corpus = build(generate(0, entities=200, relations=20, out_degree=10))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["arm"] for line in RESULTS.read_text().splitlines()} if RESULTS.exists() else set()

    for arm, overrides in ARMS.items():
        if arm in done:
            print(f"== {arm}: already recorded, skipping", flush=True)
            continue
        print(f"\n== {arm}  {overrides}", flush=True)
        start = time.time()
        cfg = TrainConfig(max_iters=iters, eval_interval=iters // 10, batch_size=512,
                          out=f"out/{arm}.pt", **overrides)
        try:
            model, history = train(corpus, cfg)
        except torch.OutOfMemoryError:
            # Another process on this desktop GPU can reclaim VRAM mid-sweep; losing one arm
            # should not lose the rest, and a missing arm must be visible rather than silent.
            print(f"== {arm}: OOM, skipping", flush=True)
            with RESULTS.open("a") as fh:
                fh.write(json.dumps({"arm": arm, **overrides, "error": "OOM"}) + "\n")
            torch.cuda.empty_cache()
            continue
        row = {"arm": arm, **overrides, "params": model.get_num_params(),
               "mins": round((time.time() - start) / 60, 1), "final": history[-1]}
        with RESULTS.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"== {arm} done: test_inferred_id={history[-1]['test_inferred_id']:.3f}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100_000)
