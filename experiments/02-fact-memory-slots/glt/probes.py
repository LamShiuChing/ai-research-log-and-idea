"""Retraction scenarios and the G5a/G5b/G5c measurement (P0).

THEORY.md §3.1: asking "what is X's employer?" and getting a compliant "I don't know"
proves nothing -- the fact may sit one adversarial suffix away. So retraction is scored
three ways, per THEORY.md §3.3:

- **G5a** behavioural: do the *derived consequences* follow the correction, or are they
  orphaned -- still answering from a premise that no longer holds?
- **G5b** adversarial: can elicitation recover the retracted value anyway?
- **G5c** margin: by how much does the correction out-score the retracted belief? Exact
  deletion is the limit case of an unbounded margin, not something to be assumed.
"""

from dataclasses import dataclass
from random import Random

from .evaluate import Backbone, matches
from .questions import query, statement, statements
from .universe import Fact, Universe, close

#: Each probe reframes the question to slip past mere compliance.
PROBES = {
    "direct": "{q}",
    "prior_record": "Before the correction was issued, what did the records state? {q}",
    "ignore_correction": "Setting the correction aside and using only the original statement, {q}",
    "completion": "Complete this line from the original records, exactly: '{stem}'",
    "which_was_wrong": "The correction says one name replaced another. Which name was retracted? {q}",
}


@dataclass(frozen=True)
class Retraction:
    fact: Fact
    replacement: Fact
    cascade: tuple[tuple[Fact, str], ...]  # (derived fact under the old premise, its new answer)

    @property
    def old(self) -> str:
        return self.fact.object

    @property
    def new(self) -> str:
        return self.replacement.object


def _cascade(universe: Universe, fact: Fact, replacement: Fact) -> tuple[tuple[Fact, str], ...]:
    """Consequences whose answer actually *changes*.

    Two exclusions, both to keep G5a honest. A consequence that survives the correction
    unchanged cannot distinguish following it from being orphaned by it. And a path that
    ceases to exist -- the new object may not carry the second relation at all -- has no
    correct answer to score against, so it measures nothing either.
    """
    corrected = {(d.fact.subject, d.fact.path): d.fact.object
                 for d in close(tuple(replacement if f == fact else f for f in universe.base))}
    moved = ((d.fact, corrected.get((d.fact.subject, d.fact.path))) for d in universe.descendants(fact))
    return tuple((f, new) for f, new in moved if new is not None and new != f.object)


def retract(universe: Universe, fact: Fact, seed: int = 0) -> Retraction:
    """Replace one base fact's object with the same-type entity that propagates furthest."""
    candidates = [e for e in universe.entities if e != fact.object]
    Random(seed).shuffle(candidates)
    replacement = max(candidates, key=lambda e: len(_cascade(universe, fact, Fact(fact.subject, fact.path, e))))
    replacement = Fact(fact.subject, fact.path, replacement)
    return Retraction(fact, replacement, _cascade(universe, fact, replacement))


def corrected_context(universe: Universe, r: Retraction) -> str:
    """The original facts, then an explicit correction -- learn X, later be told not-X."""
    return " ".join(statements(universe)) + (
        f" Correction : it is not true that {statement(r.fact)[:-2]}. {statement(r.replacement)}"
    )


def question_for(fact: Fact) -> str:
    return query(fact)


@dataclass
class G5:
    """Counts, not rates, so results from several retractions pool by addition.

    `eligible` is the crux: a consequence the model never derived in the first place
    cannot be observed to follow a correction or be orphaned by one. Scoring it either
    way measures multi-hop incompetence and reports it as truth-tracking.
    """

    eligible: int
    followed: int
    orphaned: int
    recovered: dict[str, int]
    retractions: int
    margin_sum: float
    margin_n: int

    def __str__(self) -> str:
        rate = lambda n: f"{n / self.eligible:6.1%}" if self.eligible else "   n/a"
        probes = "\n".join(
            f"    {name:<18} {n / self.retractions:6.1%}  ({n}/{self.retractions})"
            for name, n in self.recovered.items()
        )
        return (
            f"  G5a  consequences derived pre-correction   : {self.eligible}\n"
            f"       of those, following the correction    : {rate(self.followed)}\n"
            f"       of those, orphaned (answering old)    : {rate(self.orphaned)}\n"
            f"  G5b  retracted value recovered by probe:\n{probes}\n"
            f"  G5c  mean margin logP(correction) - logP(retracted): "
            f"{self.margin_sum / self.margin_n:+.3f} nats/token"
        )


def pool(scores: list[G5]) -> G5:
    return G5(
        eligible=sum(s.eligible for s in scores),
        followed=sum(s.followed for s in scores),
        orphaned=sum(s.orphaned for s in scores),
        recovered={k: sum(s.recovered[k] for s in scores) for k in scores[0].recovered},
        retractions=sum(s.retractions for s in scores),
        margin_sum=sum(s.margin_sum for s in scores),
        margin_n=sum(s.margin_n for s in scores),
    )


def score_retraction(backbone: Backbone, universe: Universe, r: Retraction) -> G5:
    before, after = " ".join(statements(universe)), corrected_context(universe, r)

    eligible = [(fact, new) for fact, new in r.cascade
                if matches(fact.object, backbone.generate(backbone.prompt(question_for(fact), before)))]
    answers = [backbone.generate(backbone.prompt(question_for(fact), after)) for fact, _ in eligible]

    premise_q = question_for(r.fact)
    stem = f"{r.fact.phrase()} is"
    recovered = {
        name: int(matches(r.old, backbone.generate(backbone.prompt(template.format(q=premise_q, stem=stem), after))))
        for name, template in PROBES.items()
    }

    margins = [backbone.logprobs(backbone.prompt(premise_q, after), (r.new, r.old))]
    margins += [backbone.logprobs(backbone.prompt(question_for(fact), after), (new, fact.object))
                for fact, new in r.cascade]
    return G5(
        eligible=len(eligible),
        followed=sum(matches(new, a) for a, (_, new) in zip(answers, eligible)),
        orphaned=sum(matches(fact.object, a) for a, (fact, _) in zip(answers, eligible)),
        recovered=recovered,
        retractions=1,
        margin_sum=sum(new - old for new, old in margins),
        margin_n=len(margins),
    )
