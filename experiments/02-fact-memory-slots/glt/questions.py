"""Questions over a universe: atomic recall (G2) and multi-hop entailment (G3).

`statements()` renders only atomic facts. A depth-2 answer is therefore never stated
beside its subject in any single sentence -- the structural guarantee behind
PROJECT_GOAL.md's evaluation trap -- while remaining derivable by composing two of them.
"""

from collections import defaultdict
from dataclasses import dataclass
from random import Random

from .universe import Fact, Universe


@dataclass(frozen=True)
class Question:
    text: str
    answer: str
    hops: int
    premises: frozenset[Fact]
    choices: tuple[str, ...]


def statement(fact: Fact) -> str:
    return f"{fact.phrase()} is {fact.object} ."


def query(fact: Fact) -> str:
    return f"{fact.phrase()} is ?"


def statements(universe: Universe) -> tuple[str, ...]:
    """Verbalised atomic facts -- the only thing a context or corpus may contain."""
    return tuple(statement(fact) for fact in universe.base)


def cooccurring(universe: Universe) -> dict[str, set[str]]:
    """Entity -> entities sharing a sentence with it, in either direction.

    Undirected on purpose. A forward edge makes the answer reachable without composing;
    a reverse edge merely puts the pair in one sentence, but for a model trained from
    scratch on exactly these sentences, co-occurrence is the dominant statistical signal
    and would be indistinguishable from having composed.

    An entity is listed as co-occurring with itself: a cycle can send a path back to its
    own subject, and "echo the subject" is a heuristic just as shallow as naming a neighbour.
    """
    adjacent: dict[str, set[str]] = defaultdict(set)
    for entity in universe.entities:
        adjacent[entity].add(entity)
    for fact in universe.base:
        adjacent[fact.subject].add(fact.object)
        adjacent[fact.object].add(fact.subject)
    return adjacent


def questions(universe: Universe, k: int = 4, seed: int = 0, per_depth: int | None = None) -> tuple[Question, ...]:
    """Forced-choice questions with `k` options, optionally sampling `per_depth` at each depth."""
    if k > len(universe.entities):
        raise ValueError(f"need >= k={k} entities for a uniform 1/k chance baseline")
    rng = Random(seed)
    facts = [(f, frozenset({f})) for f in universe.base]
    # Drop derived facts whose answer already shares a sentence with the subject: those are
    # answerable from co-occurrence alone, with no composition, so
    # scoring them would credit a shallow heuristic as multi-hop reasoning.
    direct = cooccurring(universe)
    facts += [(d.fact, d.premises) for d in universe.derived if d.fact.object not in direct[d.fact.subject]]

    if per_depth is not None:
        by_depth: dict[int, list] = {}
        for item in facts:
            by_depth.setdefault(item[0].depth, []).append(item)
        facts = [item for group in by_depth.values() for item in rng.sample(group, min(per_depth, len(group)))]

    return tuple(_question(universe, fact, premises, rng, k) for fact, premises in facts)


def _question(universe: Universe, fact: Fact, premises: frozenset[Fact], rng: Random, k: int) -> Question:
    pool = [e for e in universe.entities if e != fact.object]
    choices = rng.sample(pool, k - 1) + [fact.object]
    rng.shuffle(choices)
    return Question(query(fact), fact.object, fact.depth, premises, tuple(choices))
