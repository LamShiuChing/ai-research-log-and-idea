"""The fork must be behaviourally identical to nanoGPT before GLT changes anything.

Stronger and far cheaper than retraining: load the real shakespeare_char checkpoint into
both implementations and compare logits directly.
"""

import sys
from pathlib import Path

import pytest
import torch

from glt.model import GPT, GPTConfig

NANOGPT = Path(__file__).resolve().parents[1] / "reference" / "nanoGPT"
CKPT = NANOGPT / "out-shakespeare-char" / "ckpt.pt"

pytestmark = pytest.mark.skipif(not CKPT.exists(), reason="baseline checkpoint not on disk")


@pytest.fixture(scope="module")
def checkpoint():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    return ckpt["model_args"], state


@pytest.fixture(scope="module")
def reference_gpt():
    sys.path.insert(0, str(NANOGPT))
    import model as nanogpt  # noqa: E402  -- the upstream module, not glt.model

    return nanogpt


def test_fork_reproduces_the_baseline_checkpoint_exactly(checkpoint, reference_gpt):
    args, state = checkpoint

    ours = GPT(GPTConfig(**args))
    ours.load_state_dict(state)
    theirs = reference_gpt.GPT(reference_gpt.GPTConfig(**args))
    theirs.load_state_dict(state)
    ours.eval(), theirs.eval()

    idx = torch.randint(0, args["vocab_size"], (2, 64), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        mine, _ = ours(idx, targets=idx)
        reference, _ = theirs(idx, targets=idx)
    assert torch.equal(mine, reference)


def test_parameter_count_matches_the_recorded_baseline(checkpoint, reference_gpt):
    """PROJECT_GOAL.md §10 records 10.65M parameters for this checkpoint."""
    args, _ = checkpoint
    ours, theirs = GPT(GPTConfig(**args)), reference_gpt.GPT(reference_gpt.GPTConfig(**args))
    assert ours.get_num_params() == theirs.get_num_params()
    assert round(ours.get_num_params() / 1e6, 2) == 10.65


def test_state_dicts_have_identical_shapes(checkpoint, reference_gpt):
    args, _ = checkpoint
    ours = {k: v.shape for k, v in GPT(GPTConfig(**args)).state_dict().items()}
    theirs = {k: v.shape for k, v in reference_gpt.GPT(reference_gpt.GPTConfig(**args)).state_dict().items()}
    assert ours == theirs
