import pytest
import torch

from glt.memory import MemorySlots
from glt.model import GPT, GPTConfig

CFG = dict(vocab_size=225, block_size=8, n_layer=4, n_head=4, n_embd=64, bias=False)


def test_top_k_1_reads_exactly_one_slot():
    """The disjoint end of the sharing axis: one lookup touches one slot's values."""
    slots = MemorySlots(n_embd=64, n_slots=32, top_k=1)
    x = torch.randn(2, 5, 64)
    weight, chosen = (x @ slots.keys.T).softmax(-1).max(-1)
    assert torch.allclose(slots(x), weight.unsqueeze(-1) * slots.values[chosen], atol=1e-6)


@pytest.mark.parametrize("top_k", [1, 2, 8, 32])
def test_gradient_reaches_keys_at_every_k(top_k):
    """top_k=1 is the disjoint end and the one that matters most -- it must still learn to
    address. Normalising inside the top-k subset made this exactly zero at k=1."""
    slots = MemorySlots(n_embd=64, n_slots=32, top_k=top_k)
    slots(torch.randn(2, 5, 64)).sum().backward()
    assert slots.keys.grad.abs().sum() > 0
    assert slots.values.grad.abs().sum() > 0


def test_top_k_is_capped_at_slot_count():
    assert MemorySlots(n_embd=8, n_slots=4, top_k=99).top_k == 4


def test_read_weights_never_exceed_one():
    """Weights come from a softmax over all slots, so a top-k read is a sub-convex sum."""
    slots = MemorySlots(n_embd=16, n_slots=8, top_k=8)
    out = slots(torch.randn(3, 4, 16))
    assert out.abs().max() <= slots.values.abs().max() + 1e-6


def test_zero_slots_leaves_the_backbone_untouched():
    """n_slots=0 must stay bit-identical to nanoGPT, so the equivalence test keeps meaning."""
    torch.manual_seed(0)
    plain = GPT(GPTConfig(**CFG))
    assert all(block.slots is None for block in plain.transformer.h)
    torch.manual_seed(0)
    again = GPT(GPTConfig(**CFG, n_slots=0))
    idx = torch.randint(0, 225, (2, 7))
    plain.eval(), again.eval()
    with torch.no_grad():
        assert torch.equal(plain(idx)[0], again(idx)[0])


def test_slots_attach_to_one_block_and_add_expected_parameters():
    model = GPT(GPTConfig(**CFG, n_slots=128, slot_top_k=1))
    assert [i for i, b in enumerate(model.transformer.h) if b.slots is not None] == [2]
    baseline = GPT(GPTConfig(**CFG)).get_num_params()
    assert model.get_num_params() - baseline == 2 * 128 * CFG["n_embd"]  # keys + values


def test_slot_layer_is_selectable():
    model = GPT(GPTConfig(**CFG, n_slots=16, slot_layer=0))
    assert [i for i, b in enumerate(model.transformer.h) if b.slots is not None] == [0]


def test_dense_fast_path_matches_the_gather():
    """top_k == n_slots takes a matmul shortcut; it must be numerically the same read."""
    slots = MemorySlots(n_embd=16, n_slots=12, top_k=12)
    x = torch.randn(3, 4, 16)
    probs = (x @ slots.keys.T).softmax(-1)
    weights, idx = probs.topk(12, dim=-1)
    gathered = (weights.unsqueeze(-1) * slots.values[idx]).sum(-2)
    assert torch.allclose(slots(x), gathered, atol=1e-5)


@pytest.mark.parametrize("read", ["softmax", "gated", "threshold"])
def test_every_read_mode_passes_gradient_to_keys(read):
    slots = MemorySlots(n_embd=32, n_slots=64, top_k=8, read=read)
    slots(torch.randn(2, 5, 32)).sum().backward()
    assert slots.keys.grad.abs().sum() > 0


def test_threshold_read_can_be_exactly_silent():
    """Abstention must be expressible: softmax sums to 1 and structurally cannot be."""
    slots = MemorySlots(n_embd=32, n_slots=64, top_k=64, read="threshold")
    x = torch.randn(2, 5, 32)
    with torch.no_grad():
        slots.bias.fill_(-100.0)
    assert slots(x).abs().max() == 0.0
    assert MemorySlots(n_embd=32, n_slots=64, top_k=64, read="softmax")(x).abs().max() > 0


def test_threshold_bias_starts_at_zero_so_slots_are_not_born_dead():
    """A negative init is silent but dead: relu zeroes everything and no gradient flows."""
    slots = MemorySlots(n_embd=32, n_slots=64, top_k=64, read="threshold")
    assert slots.bias.abs().max() == 0.0
    slots(torch.randn(2, 5, 32)).sum().backward()
    assert slots.keys.grad.abs().sum() > 0
