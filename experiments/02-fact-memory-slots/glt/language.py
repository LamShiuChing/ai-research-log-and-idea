"""Language specialist: free phrasing in, a Query out (HANDOFF.md §4 step 1, extended).

THEORY.md §8 splits GLT into three specialists. This is the first. Its only job is surface
form: it holds no facts, never reads a slot, and is never shown an answer.

**Why the canonical form is the entire job.** A slot's address is a rolling hash over *every*
content token of the query (`GPT.address`, model.py). "who is Anna 's mentor ?" therefore
hashes somewhere other than the trained form `mentor of Anna is`, and one stray name anywhere
in the sentence moves the address again. Free-form text cannot address memory. This model is
the adapter: name the operation and point at the words it needs, and every measured result
downstream applies unchanged to a question asked in English.

**Why a separate model from the fact-holding backbone.** The backbone is frozen with 200 facts
in its slots, and retraining it would put the measured G2 result back in play; a parser that has
never seen a fact cannot pass a parsing test by recalling one; and §8's table gives this
specialist a job -- pattern completion over text -- that needs no memory to do.

**What makes the number falsifiable.** Question wordings and content words are both held out, so
the headline cells are phrasings never trained on and entities/relations never seen inside a
question. Training and testing on one template table would measure memorisation of that table,
which is regex with extra steps. The two noun-phrase *constructions* stay in training on
purpose: "Anna 's mentor 's manager" and "the manager of the mentor of Anna" order the same
words oppositely, and nothing in a corpus of one construction implies the other. (The holdout
covers `follow` wordings; the other shapes are trained on all of theirs.)

`head` decides how the query is produced, and it is the whole result (extract.py):

- **generate** -- ordinary language-model decoding over the vocabulary. Scores 1.000 on trained
  frames and **0.000** on any entity or relation held out of the parser's training data. The
  cause is structural, not a data shortage: a softmax over the vocabulary can only emit a token
  whose output embedding gradient has moved, so a name the parser never wrote is unreachable no
  matter how obvious the syntax makes it. Kept for the record; `follow` only.
- **point** -- fixed output slots, each a distribution over *input positions*
  ([Pointer Networks](https://arxiv.org/abs/1506.03134), Vinyals et al. 2015, which introduces
  exactly this for "variable size output dictionaries"). The answer is "position 4", and
  whatever word sits there is copied verbatim, so an unseen word costs nothing.

This is the session-2 addressing lesson a second time. Learned embeddings could not tell two
facts apart (margin 0.035, PROJECT_GOAL.md §16) and cannot name an unseen word here; both times
the fix is to address by structure -- a hash of the tokens there, an index into them here.
"""

import math
from collections import defaultdict
from itertools import product
from random import Random

import torch
import torch.nn as nn
from torch.nn import functional as F

from .corpus import PAD, Tokenizer, prefix
from .model import GPT, GPTConfig
from .universe import Fact
from .workspace import Query, count, follow, reverse, same

SEP = "=>"    # the query ends here: the pointer reads this position, generation starts after it
STOP = "is"   # the canonical form always ends here, so decoding needs no stop symbol


def possessive(subject: str, path: tuple[str, ...]) -> str:
    """`Anna 's mentor 's manager` -- subject first, relations in application order."""
    return " 's ".join((subject, *path))


def of_chain(subject: str, path: tuple[str, ...]) -> str:
    """`the manager of the mentor of Anna` -- the reverse order, which is the canonical one."""
    return " of ".join([f"the {relation}" for relation in reversed(path)] + [subject])


NOUN_PHRASES = (possessive, of_chain)

#: `follow` wordings. `{np}` is the query; `{d}` is a distractor entity that must *not* be
#: extracted. Wordings vary in verb, in clause structure, in where the query sits, and in
#: whether a competing name is present -- the axis extract.py sweeps, since a held-out wording
#: is the only thing a user actually supplies that the parser has not seen.
FRAMES = (
    "who is {np} ?",
    "what is {np} ?",
    "tell me {np} .",
    "do you know {np} ?",
    "{np} , who is it ?",
    "name {np} .",
    "i want to know {np} .",
    "can you tell me {np} ?",
    "please give me {np} .",
    "find {np} for me .",
    "{np} is who ?",
    "who would {np} be ?",
    "i know {d} , but who is {np} ?",
    "not {d} -- i mean {np} .",
    "forget {d} for a moment : who is {np} ?",
    "{np} , not {d} , is who ?",
    "give me the name of {np} .",
    "who exactly is {np} ?",
    "so , {np} ?",
    "quick question : {np} ?",
    "i forgot {np} .",
    "remind me who {np} is .",
    "let me think ... {np} ?",
    "any idea who {np} is ?",
    "is {np} known to you ?",
    "you told me about {d} earlier ; now {np} ?",
    "before you answer about {d} , {np} ?",
    "{np} -- and be quick .",
    "the answer i need is {np} .",
    "look up {np} .",
    "search for {np} please .",
    "state {np} .",
    "who , if anyone , is {np} ?",
    "i am asking about {np} , not {d} .",
    "hey , {np} ?",
    "just {np} , nothing else .",
    "could you name {np} ?",
    "would you say {np} is known ?",
    "about {np} : who ?",
    "{d} aside , who is {np} ?",
)

#: Output slots, and what each means per operation. Four, because `same` names two subjects and
#: two relations; shorter shapes park the slots they do not use on `SEP`.
#:
#:     follow    subject, relation 1, relation 2, -
#:     reverse   target,  relation,   -,          -
#:     count     target,  relation,   -,          -    (the count of that reverse)
#:     same      subject, relation,   subject,    relation
#:     none      -,       -,          -,          -    (nothing this query language expresses)
SLOTS = 4
OPS = ("follow", "reverse", "count", "same", "none")

#: Wordings per operation. `{0}..{3}` are pointer targets -- even indices are entities, odd ones
#: relations -- and `{d}` is a distractor. `none` is the reject class: questions this query
#: language cannot express, and sentences that are not questions at all. Without it the pointer
#: always points somewhere, so an unanswerable question comes back as a confident answer to a
#: different one.
SHAPES = {
    "reverse": (
        "who has {0} as their {1} ?",
        "whose {1} is {0} ?",
        "who are the people whose {1} is {0} ?",
        "list everyone whose {1} is {0} .",
        "which people have {0} as their {1} ?",
        "i want everyone whose {1} is {0} .",
        "tell me who has {0} as a {1} .",
        "not {d} -- who has {0} as their {1} ?",
        "everyone whose {1} is {0} , please .",
        "who all have {0} as a {1} ?",
    ),
    "count": (
        "how many people have {0} as their {1} ?",
        "how many have {0} as a {1} ?",
        "count everyone whose {1} is {0} .",
        "what is the number of people whose {1} is {0} ?",
        "give me the count of people whose {1} is {0} .",
        "how many people 's {1} is {0} ?",
        "tell me how many have {0} as their {1} .",
        "not {d} -- how many have {0} as a {1} ?",
    ),
    "same": (
        "is {0} 's {1} also {2} 's {3} ?",
        "are {0} 's {1} and {2} 's {3} the same person ?",
        "does {0} 's {1} equal {2} 's {3} ?",
        "is the {1} of {0} the same as the {3} of {2} ?",
        "same person : {0} 's {1} and {2} 's {3} ?",
        "do {0} 's {1} and {2} 's {3} match ?",
        "tell me if {0} 's {1} is {2} 's {3} .",
        "is {0} 's {1} the same as the {3} of {2} ?",
    ),
    "none": (
        "why is {0} 's {1} like that ?",
        "when did {0} meet {d} ?",
        "where does {0} live ?",
        "how are you ?",
        "hello there .",
        "thank you .",
        "what do you think of {0} ?",
        "is {0} nice ?",
        "tell me a story about {0} .",
        "who is the best {1} ?",
        "{0} 's {1} is {d} .",
        "i like {0} .",
        "can {0} swim ?",
        "what happened to {0} ?",
        "please explain {0} 's {1} .",
        "sorry , never mind .",
    ),
}

#: `follow` is the operation everything else composes from, so it keeps the largest share.
WEIGHTS = {"follow": 4, "reverse": 2, "count": 2, "same": 2, "none": 2}


def render(frame: str, noun_phrase, subject: str, path: tuple[str, ...], distractor: str):
    """One `follow` phrasing: the question, the canonical form, and where the words landed.

    Positions are recovered by rendering placeholders first, so they stay correct however a
    noun phrase orders its words and even when one relation appears twice in a path.
    """
    marks = tuple(f"@{i}" for i in range(1 + len(path)))
    marked = frame.format(np=noun_phrase(marks[0], marks[1:]), d=distractor).split()
    words = dict(zip(marks, (subject, *path)))
    return (" ".join(words.get(word, word) for word in marked),
            prefix(Fact(subject, path, "?")),
            tuple(marked.index(mark) for mark in marks))


def arity(frame: str) -> int:
    """How many pointer words a wording takes -- by highest index used, not by count, since a
    reject wording may name a relation without naming a subject."""
    return max((i + 1 for i in range(SLOTS) if f"{{{i}}}" in frame), default=0)


def render_shape(frame: str, words, distractor: str):
    """One phrasing of a non-`follow` shape: the question, and where its pointer words landed."""
    marks = tuple(f"@{i}" for i in range(len(words)))
    marked = frame.format(*marks, d=distractor).split()
    swap = dict(zip(marks, words))
    return (" ".join(swap.get(word, word) for word in marked),
            tuple(marked.index(mark) for mark in marks if mark in marked))


def phrasings(subject: str, path: tuple[str, ...], distractor: str, frames=FRAMES):
    """Every `follow` phrasing of one query -- the set that must collapse to one Query."""
    return [render(frame, np, subject, path, distractor)[0]
            for frame, np in product(frames, NOUN_PHRASES)]


def to_query(canonical: str) -> tuple[str, tuple[str, ...]] | None:
    """`manager of mentor of Anna is` -> `(Anna, (mentor, manager))`; None if malformed.

    Exact and not learned: the chainer walks this path one hop at a time, so a wrong split here
    would be indistinguishable from a wrong lookup. Used by the generative head only.
    """
    words = canonical.split()
    if len(words) < 4 or len(words) % 2 or words[-1] != STOP or set(words[1:-2:2]) != {"of"}:
        return None
    return words[-2], tuple(reversed(words[:-2:2]))


def split(items, held: int, seed: int = 0):
    """Deterministic (train, held-out) partition -- the same split every run, so cells compare."""
    shuffled = Random(seed).sample(tuple(items), len(items))
    return tuple(shuffled[held:]), tuple(shuffled[:held])


def vocabulary(entities, relations) -> Tokenizer:
    """Entity and relation words, plus every function word any wording of any shape can emit."""
    filler = {word for frame, np in product(FRAMES, NOUN_PHRASES)
              for word in render(frame, np, "SUBJ", ("REL",), "DIST")[0].split()}
    filler |= {word for frames in SHAPES.values() for frame in frames
               for word in render_shape(frame, ["SUBJ"] * arity(frame), "DIST")[0].split()}
    filler -= {"SUBJ", "REL", "DIST"}
    return Tokenizer(tuple(sorted(entities)) + tuple(sorted(relations))
                     + tuple(sorted(filler | {"of", STOP, SEP, PAD})))


class Pointer(nn.Module):
    """One distribution over input positions per output slot.

    The query comes from the state at `SEP`, the only position that has seen the whole question
    under a causal mask; the keys are the positions it may choose from. A slot with nothing to
    fill points at `SEP` itself, which is how a shorter shape leaves its later slots empty.
    """

    def __init__(self, n_embd: int, slots: int):
        super().__init__()
        self.slots = slots
        self.query = nn.Linear(n_embd, slots * n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, hidden, summary, sep):
        batch, time, n_embd = hidden.shape
        scores = self.query(summary).view(batch, self.slots, n_embd) @ self.key(hidden).transpose(1, 2)
        beyond = torch.arange(time, device=hidden.device) > sep[:, None]
        return (scores / math.sqrt(n_embd)).masked_fill(beyond[:, None, :], float("-inf"))


class Parser(nn.Module):
    """A small GPT with two heads: which operation this is, and which words it needs."""

    def __init__(self, config: GPTConfig, head: str = "point"):
        super().__init__()
        self.head, self.gpt = head, GPT(config)
        self.pointer = Pointer(config.n_embd, SLOTS) if head == "point" else None
        self.classify = nn.Linear(config.n_embd, len(OPS)) if head == "point" else None

    def heads(self):
        return [] if self.pointer is None else list(self.pointer.parameters()) + list(self.classify.parameters())

    def forward(self, x, targets=None, ops=None, sep=None):
        if self.head == "generate":
            return self.gpt(x, targets)
        hidden = self.gpt.hidden(x)
        summary = hidden[torch.arange(len(x), device=x.device), sep]
        logits = (self.classify(summary), self.pointer(hidden, summary, sep))
        if targets is None:
            return logits, None
        return logits, (F.cross_entropy(logits[0], ops)
                        + F.cross_entropy(logits[1].transpose(1, 2), targets))


def example(rng: Random, frames, entities, relations, depths, op: str):
    """One training row: the question, its operation, and the positions to point at."""
    if op == "follow":
        subject, distractor = rng.sample(entities, 2)
        question, _, positions = render(rng.choice(frames), rng.choice(NOUN_PHRASES), subject,
                                        tuple(rng.choices(relations, k=rng.choice(depths))), distractor)
        return question, op, positions
    frame = rng.choice(SHAPES[op])
    words = [rng.choice(entities if i % 2 == 0 else relations) for i in range(arity(frame))]
    question, positions = render_shape(frame, words, rng.choice(entities))
    return question, op, (() if op == "none" else positions)


def encode(tokenizer: Tokenizer, question: str, canonical: str) -> tuple[list[int], int]:
    """Row of ids, and the index of `SEP` -- where supervision starts and the pointer reads."""
    asked = tokenizer.encode(question)
    return asked + [tokenizer.stoi[SEP]] + tokenizer.encode(canonical), len(asked)


def pad(rows, fill: int):
    out = torch.full((len(rows), max(map(len, rows))), fill)
    for row, ids in zip(out, rows):
        row[: len(ids)] = torch.tensor(ids)
    return out


def batch(rng: Random, tokenizer, head: str, frames, entities, relations, depths,
          size: int, device: str):
    if head == "generate":
        rows = [encode(tokenizer, *render(rng.choice(frames), rng.choice(NOUN_PHRASES),
                                          *rng.sample(entities, 1),
                                          tuple(rng.choices(relations, k=rng.choice(depths))),
                                          rng.choice(entities))[:2]) for _ in range(size)]
        ids = pad([row for row, _ in rows], tokenizer.stoi[PAD])
        targets = torch.full_like(ids, -1)
        for target, (row, sep) in zip(targets, rows):
            target[sep : len(row) - 1] = torch.tensor(row[sep + 1 :])
        # contiguous(): nanoGPT's loss reshapes with .view, which a strided slice cannot do
        return (ids[:, :-1].contiguous().to(device), targets[:, :-1].contiguous().to(device), None, None)

    chosen = rng.choices(OPS, weights=[WEIGHTS[op] for op in OPS], k=size)
    examples = [example(rng, frames, entities, relations, depths, op) for op in chosen]
    rows = [tokenizer.encode(question) + [tokenizer.stoi[SEP]] for question, _, _ in examples]
    sep = torch.tensor([len(row) - 1 for row in rows])
    # An unused slot points at SEP, so a shape's arity is read off the pointers themselves.
    targets = torch.stack([torch.tensor(positions + (int(at),) * (SLOTS - len(positions)))
                           for (_, _, positions), at in zip(examples, sep)])
    ops = torch.tensor([OPS.index(op) for _, op, _ in examples])
    return (pad(rows, tokenizer.stoi[PAD]).to(device), targets.to(device),
            ops.to(device), sep.to(device))


def widest(tokenizer, entities, relations, depths) -> int:
    """Longest row any wording of any shape can produce -- the block size."""
    follows = max(len(encode(tokenizer, *render(frame, np, entities[0], (relations[0],) * max(depths),
                                                entities[1])[:2])[0])
                  for frame, np in product(FRAMES, NOUN_PHRASES))
    shapes = max(len(render_shape(frame, [entities[0]] * arity(frame), entities[1])[0].split()) + 1
                 for frames in SHAPES.values() for frame in frames)
    return max(follows, shapes)


def train(tokenizer, frames, entities, relations, head="point", depths=(1, 2), steps=4000,
          batch_size=64, n_layer=4, n_head=4, n_embd=128, lr=1e-3, device="cuda", seed=0, log=print):
    torch.manual_seed(seed)
    rng = Random(seed)
    # One hop of headroom, so a depth+1 probe fits at all: positions are learned, and those tail
    # positions get no gradient, which is a caveat on that cell rather than a bug.
    config = GPTConfig(vocab_size=len(tokenizer), n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                       bias=False, block_size=widest(tokenizer, entities, relations, (max(depths) + 1,)))
    model = Parser(config, head=head).to(device)
    optimizer = model.gpt.configure_optimizers(0.1, lr, (0.9, 0.95), device)
    if model.heads():
        optimizer.add_param_group({"params": model.heads(), "weight_decay": 0.1})
    log(f"parser[{head}]: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params  |  "
        f"vocab={len(tokenizer)}  wordings={len(frames)}  entities={len(entities)}  relations={len(relations)}")

    for step in range(steps + 1):
        _, loss = model(*batch(rng, tokenizer, head, frames, entities, relations, depths,
                               batch_size, device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % max(1, steps // 8) == 0:
            log(f"  step {step:6d}  loss {loss.item():.4f}")
    return model.eval()


def save(model: Parser, tokenizer: Tokenizer, path: str = "out/parser.pt") -> None:
    torch.save({"model": model.state_dict(), "config": model.gpt.config,
                "head": model.head, "vocab": tokenizer.itos}, path)


def load(path: str = "out/parser.pt", device: str = "cuda") -> tuple[Parser, Tokenizer]:
    """The trained parser and the tokenizer it was trained with -- they must match exactly."""
    torch.serialization.add_safe_globals([GPTConfig])
    saved = torch.load(path, map_location=device, weights_only=True)
    model = Parser(saved["config"], head=saved["head"]).to(device)
    model.load_state_dict(saved["model"])
    return model.eval(), Tokenizer(tuple(saved["vocab"]))


def assemble(op: str, words: list[str], positions, sep: int) -> Query | None:
    """Pointers back into the question -> a Query. None where the shape does not hold together.

    `none` returns None on purpose: it is the parser saying the question fits nothing this
    system can express, which is the difference between an honest refusal and a confident
    answer to a question nobody asked.
    """
    used = [p for p in positions if p != sep]
    if op == "none" or used != list(positions[: len(used)]):
        return None
    picked = [words[p] for p in used]
    if op == "follow" and len(picked) >= 2:
        return follow(picked[0], tuple(picked[1:]))
    if op in ("reverse", "count") and len(picked) == 2:
        inner = reverse(picked[0], picked[1])
        return inner if op == "reverse" else count(inner)
    if op == "same" and len(picked) == 4:
        return same(follow(picked[0], (picked[1],)), follow(picked[2], (picked[3],)))
    return None


@torch.no_grad()
def extract(model: Parser, tokenizer, questions, device: str = "cuda", max_new: int = 12):
    """The Query each question asks, or None where it fits no shape this system can express."""
    if model.head == "point":
        rows = [tokenizer.encode(question) + [tokenizer.stoi[SEP]] for question in questions]
        sep = torch.tensor([len(row) - 1 for row in rows], device=device)
        (ops, pointers), _ = model(pad(rows, tokenizer.stoi[PAD]).to(device), sep=sep)
        return [assemble(OPS[op], question.split(), positions, at)
                for op, question, positions, at in zip(ops.argmax(-1).tolist(), questions,
                                                       pointers.argmax(-1).tolist(), sep.tolist())]

    # Generative decoding is batched by question length rather than padded: the model has
    # learned absolute position embeddings, so left-padding would shift every token off the
    # distribution it trained on. `follow` only -- it has no way to name an operation.
    stop, out = tokenizer.stoi[STOP], [None] * len(questions)
    groups = defaultdict(list)
    for i, question in enumerate(questions):
        groups[len(question.split())].append(i)

    for indices in groups.values():
        ids = torch.tensor([tokenizer.encode(questions[i]) + [tokenizer.stoi[SEP]] for i in indices],
                           device=device)
        start = ids.shape[1]
        for _ in range(min(max_new, model.gpt.config.block_size - start)):
            logits, _ = model(ids)
            ids = torch.cat((ids, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
            if (ids[:, start:] == stop).any(dim=1).all():
                break
        for i, row in zip(indices, ids[:, start:].tolist()):
            decoded = tokenizer.decode(row[: row.index(stop) + 1] if stop in row else row)
            parsed = to_query(decoded)
            out[i] = follow(*parsed) if parsed else None
    return out
