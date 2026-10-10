import pytest

from glt.probes import PROBES, corrected_context, question_for, retract
from glt.questions import statement
from glt.universe import generate

U = generate(0, entities=60, relations=12, out_degree=6)
R = retract(U, U.base[0])


def test_replacement_contradicts_exactly_one_atomic_fact():
    assert R.replacement.subject == R.fact.subject and R.replacement.path == R.fact.path
    assert R.new != R.old


def test_cascade_entries_all_actually_change():
    """A consequence that survives unchanged would count as both followed and orphaned."""
    assert R.cascade and all(new != fact.object for fact, new in R.cascade)


def test_cascade_excludes_paths_that_cease_to_exist():
    """The new object may not carry the second relation, leaving no answer to score."""
    assert all(new is not None for _, new in R.cascade)


def test_cascade_is_a_subset_of_the_descendants():
    assert {f for f, _ in R.cascade} <= {d.fact for d in U.descendants(R.fact)}


def test_dense_graph_gives_the_cascade_real_power():
    """The chain design gave 2 consequences per retraction; G5a needs more than that."""
    assert len(R.cascade) >= 5


def test_context_states_the_old_fact_then_corrects_it():
    head, _, correction = corrected_context(U, R).partition("Correction :")
    assert statement(R.fact) in head
    assert R.old in correction and R.new in correction
    assert correction.index(R.old) < correction.index(R.new)


def test_every_probe_formats():
    stem = f"{R.fact.phrase()} is"
    assert all(t.format(q=question_for(R.fact), stem=stem) for t in PROBES.values())


def test_probes_never_leak_the_retracted_name():
    """A probe naming the old value would measure echoing, not recovery."""
    stem = f"{R.fact.phrase()} is"
    assert all(R.old not in t.format(q=question_for(R.fact), stem=stem) for t in PROBES.values())


@pytest.mark.parametrize("seed", range(3))
def test_deterministic_and_nondegenerate_across_seeds(seed):
    u = generate(seed, entities=40, relations=10, out_degree=5)
    for fact in u.base[:6]:
        r = retract(u, fact, seed)
        assert r == retract(u, fact, seed)
        assert all(new != f.object for f, new in r.cascade)
