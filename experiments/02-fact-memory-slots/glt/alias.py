"""Bind the words you actually use to the symbols the memory addresses.

The memory holds facts about symbols -- 200 entity tokens and 20 relation tokens it was trained
to read and write. What those symbols are *called* is cosmetic, which lexicon.py established
when it relabelled them as names for the demo. This makes the relabelling dynamic: the first
time you mention your sister, "sister" is bound to a free relation symbol and "Mia" to a free
entity symbol, and from then on the memory can store, recall, delete and reason about them with
every measured guarantee intact.

Nothing here is learned. It is a dictionary, and the exactness of the whole system depends on it
staying one: a word maps to exactly one symbol, forever, or a fact written under one name would
be unreachable under another. Every spelling of a word is registered against the same symbol for
the same reason -- a language model writes "my sisters boss" as readily as "my sister's".

**Only teaching creates a binding.** A question may only mention words already bound. That is
what stops a language model from inventing a person: an unknown name in a question is reported
as unknown rather than quietly given a symbol and an empty slot to answer from.
"""

import json
import re
from pathlib import Path

ENTITY, RELATION = "entity", "relation"
#: Dropped before binding, so "my sister", "the sister" and "sister" are one relation.
NOISE = re.compile(r"^(my|your|our|their|his|her|its|the|a|an)\s+|'s$|s'$", re.I)


def normalise(word: str) -> str:
    """Lowercase, strip a leading article or possessive, strip a trailing 's."""
    return NOISE.sub("", word.strip().lower()).strip()


def variants(word: str) -> list[str]:
    """The forms one word arrives in, canonical first. Two spellings of one relation must never
    claim two symbols -- a fact written under one would be unreachable under the other."""
    base = normalise(word)
    return [base, base[:-1]] if base.endswith("s") and len(base) > 3 else [base]


class Alias:
    """Two-way map between the words you use and the symbols the memory understands."""

    def __init__(self, entities, relations, path: str = "out/alias.json"):
        self.pool = {ENTITY: list(entities), RELATION: list(relations)}
        self.path, self.symbols, self.words, self.kinds = Path(path), {}, {}, {}
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            for word, (symbol, kind) in saved["bindings"].items():
                self._claim(word, symbol, kind)
            self.words.update(saved["display"])

    def _claim(self, word: str, symbol: str, kind: str) -> str:
        """Register one spelling. Several spellings may share a symbol, so taking it is idempotent."""
        if symbol in self.pool[kind]:
            self.pool[kind].remove(symbol)
        self.symbols[word], self.kinds[word] = symbol, kind
        self.words.setdefault(symbol, word)
        return symbol

    def bind(self, word: str, kind: str) -> str | None:
        """The symbol for a word, claiming a free one the first time. None if the pool is full."""
        found = self.symbol(word)
        if found or not self.pool[kind]:
            return found
        symbol = self.pool[kind][0]
        for form in variants(word):
            self._claim(form, symbol, kind)
        # Keyed on the normalised form, shown as you wrote it: "Mia" is a name, not "mia".
        self.words[symbol] = NOISE.sub("", word.strip()).strip()
        return symbol

    def symbol(self, word: str) -> str | None:
        """Look up without binding -- what a *question* is allowed to do."""
        return next((self.symbols[form] for form in variants(word) if form in self.symbols), None)

    def word(self, symbol) -> str:
        return self.words.get(symbol, symbol)

    def known(self, kind: str) -> list[str]:
        """The canonical spelling of everything bound, for the `what do you know` command."""
        return sorted({self.words[symbol] for word, symbol in self.symbols.items()
                       if self.kinds[word] == kind})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pairs = {word: [symbol, self.kinds[word]] for word, symbol in self.symbols.items()}
        self.path.write_text(json.dumps({"bindings": pairs, "display": self.words}, indent=1),
                             encoding="utf-8")
