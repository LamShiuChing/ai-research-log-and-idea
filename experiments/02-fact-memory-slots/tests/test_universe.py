from collections import Counter

import pytest

from glt.universe import GROKKING_PHI, Fact, close, generate

U = generate(0, entities=60, relations=12, out_degree=6)


def test_derived_facts_are_never_asserted():
    """The load-bearing property: if a derived answer were stated, G3 measures recall."""
    assert {d.fact for d in U.derived}.isdisjoint(U.base)


def test_premises_are_atomic_and_resufficient():
    for d in U.derived:
        assert d.premises <= set(U.base)
        assert d.fact in {r.fact for r in close(tuple(d.premises))}


def test_depth_matches_path_length_and_premise_count():
    for d in U.derived:
        assert d.fact.depth == len(d.fact.path) == len(d.premises) == 2


def test_every_question_has_exactly_one_answer():
    keys = Counter((f.subject, f.path) for f in U.base + tuple(d.fact for d in U.derived))
    assert max(keys.values()) == 1


def test_phrase_nests_outermost_relation_first():
    fact = Fact("zorbin", ("keeper", "warden"), "marlowe")
    assert fact.phrase() == "warden of keeper of zorbin"


def test_descendants_are_the_retraction_cascade():
    premise = U.base[0]
    assert set(U.descendants(premise)) == {d for d in U.derived if premise in d.premises}
    assert U.descendants(premise)


def test_phi_equals_out_degree_and_lands_in_the_grokking_band():
    """Wang et al. (arXiv:2405.15071): phi drives grokking, dataset size does not."""
    for k in (6, 10, 18):
        u = generate(0, entities=50, relations=20, out_degree=k)
        assert u.phi == k and GROKKING_PHI[0] <= u.phi <= GROKKING_PHI[1]


def test_phi_is_invariant_to_entity_count():
    assert {generate(0, entities=n, relations=20, out_degree=10).phi for n in (40, 120, 300)} == {10.0}


def test_words_are_distinct_and_non_overlapping():
    """One name being a substring of another would break word-level tokenisation."""
    words = list(U.entities) + list(U.relations)
    assert len(words) == len(set(words))
    assert not [(a, b) for a in words for b in words if a != b and a in b]


def test_deterministic_by_seed():
    assert generate(1, entities=40) == generate(1, entities=40) != generate(2, entities=40)


def test_no_self_loops():
    """A self-loop makes the answer recoverable from the subject alone."""
    assert not [f for f in U.base if f.subject == f.object]


def test_closure_composes_a_two_cycle_without_spinning():
    facts = (Fact("a", ("r",), "b"), Fact("b", ("s",), "a"))
    assert {d.fact for d in close(facts, max_depth=2)} == {
        Fact("a", ("r", "s"), "a"), Fact("b", ("s", "r"), "b")}


@pytest.mark.parametrize("seed", range(3))
def test_invariants_hold_across_seeds(seed):
    u = generate(seed, entities=40, relations=10, out_degree=5)
    assert {d.fact for d in u.derived}.isdisjoint(u.base)
    assert all(d.premises <= set(u.base) for d in u.derived)
