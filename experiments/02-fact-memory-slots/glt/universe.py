"""Synthetic fact universes with a ground-truth entailment graph (P0).

Entities and relations are nonsense words, so no fact can occur in any pretraining
corpus. A fact is a *relation path* applied to a subject, rendered as nested `of`
phrases -- "warden of keeper of Zorbin" -- which composes to any depth over a tiny
word-level vocabulary.

The design target is the **inferred-to-atomic ratio** phi, which Wang et al. (NeurIPS
2024, arXiv:2405.15071) find drives grokking of implicit multi-hop reasoning, and which
is "not influenced by the size of training data". Here phi is set by `out_degree` alone:
each atomic edge extends by every out-edge of its object, so phi == out_degree at depth 2.
Scaling entity count does not move it.
"""

from collections import defaultdict
from dataclasses import dataclass
from random import Random

ONSETS = "b d f g h k l m n p r s t v z br dr fl gr kl kr pl sk st tr vr".split()
NUCLEI = "a e i o u a e i o u ae ei or ur".split()
CODAS = ["", "", "l", "n", "r", "s", "th", "x", "sh", "ld"]

#: phi lands in the 5.4-18.0 band Wang et al. sweep; 10 sits mid-range.
GROKKING_PHI = (5.4, 18.0)

#: The answer to a question with no fact. Silence has to be a token the model can emit, or
#: "I don't know" is not in its output space at all.
NONE = "<none>"


@dataclass(frozen=True)
class Fact:
    """`path` is in application order: path[0] is applied to `subject` first."""

    subject: str
    path: tuple[str, ...]
    object: str

    @property
    def depth(self) -> int:
        return len(self.path)

    def phrase(self) -> str:
        """Nested rendering: path ('keeper', 'warden') reads 'warden of keeper of X'."""
        return " of ".join((*reversed(self.path), self.subject))


@dataclass(frozen=True)
class Derivation:
    """A fact plus the atomic facts it rests on. Atomic facts are their own premise."""

    fact: Fact
    premises: frozenset[Fact]


@dataclass(frozen=True)
class Universe:
    seed: int
    entities: tuple[str, ...]
    relations: tuple[str, ...]
    base: tuple[Fact, ...]
    derived: tuple[Derivation, ...]

    @property
    def phi(self) -> float:
        """Inferred-to-atomic ratio -- the quantity that governs grokking speed."""
        return len(self.derived) / len(self.base)

    def descendants(self, premise: Fact) -> tuple[Derivation, ...]:
        """Everything entailed by `premise` -- the retraction cascade G5a must withdraw."""
        return tuple(d for d in self.derived if premise in d.premises)


def absences(universe: "Universe") -> tuple[Fact, ...]:
    """Every (subject, relation) pair the universe has no edge for, answered by silence.

    Each entity gets `out_degree` of the relations, so half of all pairs are questions with no
    answer. A model never shown one cannot learn that silence is an option and answers with its
    best guess instead -- measured at 22% confident confabulation before this existed.
    """
    edges = {(fact.subject, fact.path[0]) for fact in universe.base}
    return tuple(Fact(subject, (relation,), NONE)
                 for subject in universe.entities for relation in universe.relations
                 if (subject, relation) not in edges)


def _word(rng: Random) -> str:
    return "".join(
        rng.choice(ONSETS) + rng.choice(NUCLEI) + rng.choice(CODAS)
        for _ in range(rng.choices((2, 3), (4, 1))[0])
    )


def _distinct_words(rng: Random, n: int, used: set[str]) -> tuple[str, ...]:
    words: list[str] = []
    while len(words) < n:
        word = _word(rng)
        if word not in used:
            used.add(word)
            words.append(word)
    return tuple(words)


def close(base: tuple[Fact, ...], max_depth: int = 2) -> tuple[Derivation, ...]:
    """Extend every fact by the out-edges of its object, to `max_depth`.

    (subject, path) determines a unique object because each entity has at most one
    out-edge per relation -- so every question has exactly one correct answer.
    """
    out = defaultdict(list)
    for fact in base:
        out[fact.subject].append(fact)

    derived: dict[Fact, frozenset[Fact]] = {}
    frontier = {fact: frozenset({fact}) for fact in base}
    for _ in range(max_depth - 1):
        nxt = {
            Fact(fact.subject, fact.path + edge.path, edge.object): premises | {edge}
            for fact, premises in frontier.items()
            for edge in out[fact.object]
        }
        frontier = {f: p for f, p in nxt.items() if f not in derived and f not in frontier}
        derived.update(frontier)
    return tuple(Derivation(f, p) for f, p in derived.items())


def generate(seed: int, entities: int = 500, relations: int = 20, out_degree: int = 10,
             max_depth: int = 2) -> Universe:
    """A dense relation graph: each entity gets `out_degree` distinct outgoing relations.

    phi == out_degree at max_depth 2, so `out_degree` is the grokking knob.
    """
    rng = Random(seed)
    used: set[str] = set()
    names = _distinct_words(rng, entities, used)
    rels = _distinct_words(rng, relations, used)
    # Uniform over every entity except the subject: a self-loop would make a question
    # answerable from its own subject, and would let one edge serve as both premises.
    base = tuple(
        Fact(subject, (relation,), names[(i + 1 + rng.randrange(entities - 1)) % entities])
        for i, subject in enumerate(names)
        for relation in rng.sample(rels, out_degree)
    )
    return Universe(seed, names, rels, base, close(base, max_depth))
