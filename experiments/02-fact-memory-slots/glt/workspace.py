"""The shared board, the query language, and the control loop -- THEORY.md §8's workspace.

Three specialists, no wiring between them. Each one only reads the board and writes to it:

    language   English <-> a Query                             (glt/language.py)
    memory     `follow` one hop, or `scan` for the reverse     (injected; `Slots` in chat.py)
    logic      decompose, count, compare, spot contradictions  (`infer`, `conflicts`, here)

**The query language.** A `Query` is one question shape. `follow` and `reverse` reach memory;
`count` and `same` wrap other queries and are ordinary code over their answers. Adding an
operation means adding a shape here and a few lines in `infer` -- deliberately, because an
operation that stays exact is worth more than one a network approximates.

    follow    Anna's mentor's manager       one lookup per hop, chained
    reverse   who has Anna as their mentor  every entity whose relation points at Anna
    count     how many                      len() of the answer it wraps
    same      is X's r also Y's s           == of the two answers it wraps

**Why `reverse` scans instead of using an index.** A slot's address is a hash of (relation,
subject), so a question that fixes the *object* cannot compute an address. An inverse index
would answer it instantly and would also be an external fact table -- exactly the line §8 says
must never be crossed, since facts have to stay in network parameters. Scanning keeps them
there: it asks the ordinary forward question of every entity at once, in one batched pass, and
keeps the ones that point at the target. Exact, parametric, and O(entities) -- a real ceiling at
large entity counts, and a reason to revisit this rather than a reason not to build it.

**Contradiction.** Two notes may answer one question differently -- memory says one thing, you
assert another. The board keeps both and `conflicts` reports them, instead of letting the later
one win silently. `resolve` then does the [Doyle 1979](https://www.sciencedirect.com/science/article/abs/pii/0004370279900080)
move: keep the better-supported answer and withdraw everything resting on the loser.

The control rule is §8's proposal -- run every specialist that can fire, halt when a cycle adds
nothing new. Termination is structural: `infer` only ever posts sub-queries that are strictly
smaller, and duplicate notes are dropped.

Deliberately free of torch: memory enters as any object with `follow(relation, subject)` and
`scan(relation, target)`, so the loop is testable without a GPU and the specialist is swappable.
"""

from collections import defaultdict
from dataclasses import dataclass

from .universe import Fact

ASKED, MEMORY, DERIVED, TOLD = "asked", "memory", "derived", "told"


@dataclass(frozen=True)
class Query:
    """One question. `follow`/`reverse` name a subject and a path; `count`/`same` wrap queries."""

    op: str
    subject: str = ""
    path: tuple[str, ...] = ()
    inner: tuple["Query", ...] = ()

    def __str__(self) -> str:
        if self.op == "follow":
            return " of ".join((*reversed(self.path), self.subject))
        if self.op == "reverse":
            return f"whose {self.path[0]} is {self.subject}"
        return f"{self.op} of ({'; '.join(map(str, self.inner))})"


def follow(subject: str, path) -> Query:
    return Query("follow", subject, tuple(path))


def reverse(target: str, relation: str) -> Query:
    return Query("reverse", target, (relation,))


def count(inner: Query) -> Query:
    return Query("count", inner=(inner,))


def same(left: Query, right: Query) -> Query:
    return Query("same", inner=(left, right))


@dataclass(frozen=True)
class Note:
    """One line on the board: a question, its answer once it has one, and what produced it.

    `premises` are the atomic facts it rests on -- a fact read from memory is its own premise,
    following universe.Derivation, which is what makes `retract` a set operation rather than a
    research problem. `answer` is None while the question is still open.
    """

    query: Query
    answer: object = None
    source: str = ASKED
    premises: tuple[Fact, ...] = ()
    confidence: float = 1.0

    @property
    def open(self) -> bool:
        return self.answer is None

    @property
    def grounds(self) -> tuple[Fact, ...]:
        if self.premises or self.open or self.query.op != "follow":
            return self.premises
        return (Fact(self.query.subject, self.query.path, self.answer),)

    def __str__(self) -> str:
        shown = ", ".join(self.answer) if isinstance(self.answer, tuple) else self.answer
        return f"{self.query} is {'?' if self.open else shown}"


def best(notes) -> Note | None:
    """The best-supported claim. Ties go to the later one: AGM gives priority to new information."""
    return max(reversed(list(notes)), key=lambda note: note.confidence, default=None)


def answer_to(board, query: Query) -> Note | None:
    """The best-supported answer to a question, or None while it is unanswered."""
    return best(note for note in board if note.query == query and not note.open)


def size(answer) -> int:
    """One name is one thing; a set is as many as it holds."""
    return len(answer) if isinstance(answer, tuple) else 1


def recall(board, memory) -> list[Note]:
    """Memory specialist: one forward pass per open lookup, one batched pass for a reverse."""
    out = []
    for note in board:
        query = note.query
        if not note.open or answer_to(board, query) is not None:
            continue
        if query.op == "follow" and len(query.path) == 1:
            word, confidence = memory.follow(query.path[0], query.subject)
            out.append(Note(query, word, MEMORY, confidence=confidence))
        elif query.op == "reverse":
            found = memory.scan(query.path[0], query.subject)
            out.append(Note(query, tuple(subject for subject, _ in found), MEMORY,
                            tuple(Fact(subject, query.path, query.subject) for subject, _ in found),
                            min((c for _, c in found), default=1.0)))
    return out


def wrapper(board, query: Query, combine) -> list[Note]:
    """Post whichever sub-queries are unanswered, or close the note once they all have answers."""
    parts = [answer_to(board, inner) for inner in query.inner]
    if any(part is None for part in parts):
        return [Note(inner, source=DERIVED) for inner, part in zip(query.inner, parts) if part is None]
    return [Note(query, combine(*parts), DERIVED,
                 tuple(fact for part in parts for fact in part.grounds),
                 min(part.confidence for part in parts))]


def infer(board) -> list[Note]:
    """Logic specialist: split a multi-hop path, or close a wrapper once its parts have landed."""
    out = []
    for note in board:
        query = note.query
        if not note.open or answer_to(board, query) is not None:
            continue
        if query.op == "count":
            out += wrapper(board, query, lambda part: size(part.answer))
        elif query.op == "same":
            out += wrapper(board, query, lambda left, right: left.answer == right.answer)
        elif query.op == "follow" and len(query.path) > 1:
            first = answer_to(board, follow(query.subject, query.path[:1]))
            if first is None:
                out.append(Note(follow(query.subject, query.path[:1]), source=DERIVED))
                continue
            rest = follow(first.answer, query.path[1:])
            closing = answer_to(board, rest)
            out.append(Note(rest, source=DERIVED) if closing is None else
                       Note(query, closing.answer, DERIVED, first.grounds + closing.grounds,
                            min(first.confidence, closing.confidence)))
    return out


def run(board: list[Note], memory) -> list[Note]:
    """Cycle both specialists until nothing new appears. The board is returned, not mutated.

    Notes are keyed by question *and* answer, so a second, different answer to a settled
    question lands on the board as a contradiction instead of being dropped as a duplicate.
    """
    board = list(board)
    while True:
        seen = {(note.query, str(note.answer)) for note in board}
        fresh = {}
        for note in recall(board, memory) + infer(board):
            if (note.query, str(note.answer)) not in seen:
                fresh.setdefault((note.query, str(note.answer)), note)
        if not fresh:
            return board
        board += fresh.values()


def solve(query: Query, memory, board=()) -> tuple[list[Note], Note | None]:
    """Ask one question against a board, returning the whole board and the closing note."""
    already = any(note.query == query for note in board)
    settled = run(list(board) if already else [*board, Note(query)], memory)
    return settled, answer_to(settled, query)


def retract(board, subject: str, path: tuple[str, ...], answer=None) -> list[Note]:
    """Drop the fact answering (subject, path) and everything resting on it.

    Identified by the question, because that is what a correction withdraws: "her mentor is not
    Ben after all" invalidates the slot, not one word. `answer` narrows it to one *version* of
    that fact, which is what a contradiction needs -- withdrawing the losing value must not take
    the winning one with it.

    THEORY.md §8: with premises recorded, the retraction cascade is bookkeeping rather than a
    gradient problem -- invalidating a premise deletes its descendants by construction.
    """
    def rests_on(fact):
        return (fact.subject == subject and fact.path == path
                and (answer is None or fact.object == answer))

    return [note for note in board if not any(map(rests_on, note.grounds))]


def told(query: Query, answer, confidence: float = 1.0) -> Note:
    """An assertion from outside -- the second source that makes a contradiction possible."""
    return Note(query, answer, TOLD, confidence=confidence)


def conflicts(board) -> list[tuple[Note, ...]]:
    """Questions the board answers two ways -- the contradiction a silent overwrite would hide."""
    answers = defaultdict(list)
    for note in board:
        if not note.open:
            answers[note.query].append(note)
    return [tuple(notes) for notes in answers.values() if len({str(n.answer) for n in notes}) > 1]


def resolve(board) -> tuple[list[Note], list[Note]]:
    """Keep the best-supported answer to each contradiction; withdraw what the losers rested on.

    Doyle's move: repair a contradiction by retracting the assumptions behind it rather than by
    editing the conclusion. Returns the repaired board and the notes it dropped.
    """
    kept = list(board)
    for group in conflicts(board):
        winner = best(group)
        for loser in (note for note in group if note is not winner):
            for fact in loser.grounds:
                kept = retract(kept, fact.subject, fact.path, fact.object)
            kept = [note for note in kept if note is not loser]
    return kept, [note for note in board if not any(note is k for k in kept)]


def trace(board, note: Note) -> list[Note]:
    """The lookups behind an answer -- what `where` prints, and what a retraction removes."""
    grounds = set(note.grounds)
    return [other for other in board if other.query != note.query and not other.open
            and other.query.op == "follow" and set(other.grounds) <= grounds]
