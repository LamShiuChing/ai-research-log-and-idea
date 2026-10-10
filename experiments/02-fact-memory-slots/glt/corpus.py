"""Word-level tokenizer and training corpus, with the ID/OOD split G3 needs.

One token per entity and per relation, plus `of is . ?`. Vocabulary stays in the
hundreds, so the model spends its capacity on structure rather than spelling.

The split follows Wang et al. (arXiv:2405.15071), who hold out 5% of atomic facts as
OOD and report that grokked composition generalizes to held-out *ID* inferred facts but
**not** to OOD ones. Our reading of that protocol:

- every atomic fact is in the training stream -- the model must know them all;
- an inferred fact is OOD if any premise is an OOD atomic fact, i.e. that atomic fact
  never appears *inside a composed training example*, only on its own;
- ID inferred facts are split into a training share and a held-out share.

So `test_inferred_id` measures whether the composition rule was learned, and
`test_inferred_ood` whether it generalizes to facts it was never composed over.
Evaluation sets exclude co-occurrence shortcuts (questions.py), which are answerable
without composing at all.
"""

from dataclasses import dataclass
from random import Random

import numpy as np

from .questions import cooccurring, statement
from .universe import NONE, Fact, Universe

PAD = "<pad>"
#: NONE goes last so every earlier id keeps its value and checkpoints from before it still load.
FUNCTION_WORDS = ("of", "is", ".", "?", PAD, NONE)


class Tokenizer:
    """Whitespace word-level. Entities and relations are single tokens by construction."""

    def __init__(self, words: tuple[str, ...]):
        self.itos = words
        self.stoi = {word: i for i, word in enumerate(words)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[word] for word in text.split()]

    def decode(self, ids) -> str:
        return " ".join(self.itos[int(i)] for i in ids)


@dataclass(frozen=True)
class Corpus:
    tokenizer: Tokenizer
    train: np.ndarray
    heldout: dict[str, tuple[Fact, ...]]
    ood_atomic: frozenset[Fact]  # never composed inside a training example

    def __str__(self) -> str:
        held = "  ".join(f"{name}={len(facts)}" for name, facts in self.heldout.items())
        return f"vocab={len(self.tokenizer)}  train_rows={len(self.train):,}x{self.train.shape[1]}  {held}"


def build(universe: Universe, seed: int = 0, ood_frac: float = 0.05,
          inferred_train_frac: float = 0.7) -> Corpus:
    rng = Random(seed)
    tokenizer = Tokenizer(tuple(sorted(universe.entities)) + tuple(sorted(universe.relations)) + FUNCTION_WORDS)

    ood_atomic = set(rng.sample(universe.base, round(len(universe.base) * ood_frac)))
    id_inferred = [d for d in universe.derived if d.premises.isdisjoint(ood_atomic)]
    ood_inferred = [d for d in universe.derived if not d.premises.isdisjoint(ood_atomic)]

    rng.shuffle(id_inferred)
    split = round(len(id_inferred) * inferred_train_frac)
    train_inferred, test_inferred_id = id_inferred[:split], id_inferred[split:]

    sentences = [statement(f) for f in universe.base] + [statement(d.fact) for d in train_inferred]
    rng.shuffle(sentences)

    # One sentence per row, right-padded. Packing sentences into a continuous stream instead
    # would place a given fact at an arbitrary position each time it is sampled, while
    # evaluation always feeds a prefix from position 0 -- with learned absolute position
    # embeddings that is an off-distribution eval, and the facts do not get memorised.
    encoded = [tokenizer.encode(s) for s in sentences]
    width = max(map(len, encoded))
    rows = np.full((len(encoded), width), tokenizer.stoi[PAD], dtype=np.uint16)
    for row, ids in zip(rows, encoded):
        row[: len(ids)] = ids

    adjacent = cooccurring(universe)
    composable = lambda ds: tuple(d.fact for d in ds if d.fact.object not in adjacent[d.fact.subject])
    return Corpus(
        ood_atomic=frozenset(ood_atomic),
        tokenizer=tokenizer,
        train=rows,
        heldout={
            "train_atomic": universe.base,
            "train_inferred": tuple(d.fact for d in train_inferred),
            "test_inferred_id": composable(test_inferred_id),
            "test_inferred_ood": composable(ood_inferred),
        },
    )


def prefix(fact: Fact) -> str:
    """Everything up to the answer: the model's next token should be `fact.object`."""
    return f"{fact.phrase()} is"
