from collections import Counter

import pytest

from glt.questions import cooccurring, questions, statement, statements
from glt.universe import generate

U = generate(0, entities=60, relations=12, out_degree=6)
QS = questions(U)
CTX = statements(U)


def test_multi_hop_answers_are_never_stated_beside_their_subject():
    """The evaluation trap: no single sentence may pair a subject with a derived answer,
    which would let a shallow neighbour-naming heuristic score as composition."""
    direct = cooccurring(U)
    for q in (q for q in QS if q.hops > 1):
        subject = q.text.split(" of ")[-1].removesuffix(" is ?")
        assert q.answer not in direct[subject]
        assert not [s for s in CTX if subject in s.split() and q.answer in s.split()]


def test_context_verbalises_atomic_facts_only():
    assert len(CTX) == len(U.base)
    assert all(sentence.count(" of ") == 1 for sentence in CTX)


def test_statement_and_query_share_a_phrase():
    fact = U.base[0]
    assert statement(fact).startswith(fact.phrase()) and statement(fact).endswith(f"is {fact.object} .")


def test_answer_is_among_choices_and_chance_is_uniform():
    assert all(q.answer in q.choices and len(set(q.choices)) == 4 for q in QS)


def test_covers_both_depths_minus_shortcuts():
    direct = cooccurring(U)
    composable = [d for d in U.derived if d.fact.object not in direct[d.fact.subject]]
    assert dict(Counter(q.hops for q in QS)) == {1: len(U.base), 2: len(composable)}
    assert 0 < len(U.derived) - len(composable) < len(U.derived) * 0.2


def test_premises_link_questions_to_the_retraction_cascade():
    assert all(q.premises <= set(U.base) and len(q.premises) == q.hops for q in QS)


def test_per_depth_sampling_balances_the_set():
    sampled = questions(U, per_depth=25)
    assert dict(Counter(q.hops for q in sampled)) == {1: 25, 2: 25}


def test_rejects_universes_too_small_for_a_uniform_baseline():
    with pytest.raises(ValueError, match="chance baseline"):
        questions(generate(0, entities=3, relations=4, out_degree=2), k=4)


def test_deterministic_by_seed():
    assert questions(U, seed=1) == questions(U, seed=1) != questions(U, seed=2)
