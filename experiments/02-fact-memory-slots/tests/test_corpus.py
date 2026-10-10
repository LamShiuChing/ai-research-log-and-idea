from glt.corpus import FUNCTION_WORDS, PAD, build, prefix
from glt.questions import statement
from glt.universe import generate

U = generate(0, entities=80, relations=12, out_degree=6)
C = build(U)
TRAIN_SENTENCES = {C.tokenizer.decode(row).replace(f" {PAD}", "").strip() for row in C.train}


def test_held_out_facts_never_appear_in_training():
    """The whole point of the split: a test fact seen in training measures memorisation."""
    for name in ("test_inferred_id", "test_inferred_ood"):
        assert not {statement(f) for f in C.heldout[name]} & TRAIN_SENTENCES, name


def test_every_atomic_fact_is_in_training():
    """The model must know the atomic facts; only their compositions are held out."""
    assert {statement(f) for f in U.base} <= TRAIN_SENTENCES


def test_ood_split_is_defined_by_premises():
    """OOD means: at least one premise never appeared inside a composed training example."""
    premises = {d.fact: d.premises for d in U.derived}
    assert C.ood_atomic
    assert all(premises[f] & C.ood_atomic for f in C.heldout["test_inferred_ood"])
    assert all(not (premises[f] & C.ood_atomic) for f in C.heldout["test_inferred_id"])


def test_entities_and_relations_are_single_tokens():
    assert all(len(C.tokenizer.encode(word)) == 1 for word in U.entities + U.relations)


def test_vocab_is_exactly_the_universe_plus_function_words():
    assert len(C.tokenizer) == len(U.entities) + len(U.relations) + len(FUNCTION_WORDS)


def test_tokenizer_round_trips():
    text = statement(U.base[0])
    assert C.tokenizer.decode(C.tokenizer.encode(text)) == text


def test_prefix_stops_exactly_before_the_answer():
    fact = U.base[0]
    assert statement(fact) == f"{prefix(fact)} {fact.object} ."
    assert fact.object not in prefix(fact).split()


def test_rows_are_padded_to_a_uniform_width():
    assert C.train.shape[1] == max(len(C.tokenizer.encode(s)) for s in TRAIN_SENTENCES)
    assert (C.train[:, :2] != C.tokenizer.stoi[PAD]).all()  # padding is right-hand only


def test_deterministic_by_seed():
    a, b = build(U, seed=1), build(U, seed=2)
    assert (build(U, seed=1).train == a.train).all()
    assert a.train.shape != b.train.shape or (a.train != b.train).any()
