"""The memory specialist as a dict -- the baseline the slot store has to beat.

The board never sees a token, a slot or a tensor. It asks for one hop and for who points at
what (`glt/workspace.py:134,137`), so the whole upper stack -- parser, board, provenance,
retraction -- is already substrate-independent and has simply never been run on a second
substrate. This is that second substrate.

Two things it settles at once:

- **the ceiling.** `chat.Slots` answers with an argmax over a 226-token vocabulary, so it can
  hold 200 entities and 20 relations, ever. A dict has no vocabulary and no ceiling.
- **the question.** A dict is exact, unbounded, instant and free. Anything the slot store
  cannot do *better* than this is not a reason for the slot store to exist. `parity.py` runs
  the G5a suite on both.

What a dict cannot do, and what the slot store therefore has to earn back: abstain on a
question it was never taught, answer with a confidence below 1.0, score whether a fact is worth
storing at all (`glt/admission.py`), or answer a fact nobody ever stated.
"""


class Facts:
    """A dict behind the memory specialist's interface. Every answer is certain, by construction."""

    def __init__(self, facts=()):
        self.facts = dict(facts)

    def follow(self, relation, subject):
        """One hop. `None` is "no such fact", which the board reads as a still-open question."""
        return self.facts.get((relation, subject)), 1.0

    def scan(self, relation, target):
        """Everyone whose `relation` points at `target` -- the reverse lookup, exactly."""
        return [(subject, 1.0) for (rel, subject), obj in self.facts.items()
                if (rel, obj) == (relation, target)]

    def write(self, relation, subject, obj) -> None:
        self.facts[relation, subject] = obj

    def erase(self, relation, subject):
        """The object that was there, so a caller can put it back."""
        return self.facts.pop((relation, subject), None)
