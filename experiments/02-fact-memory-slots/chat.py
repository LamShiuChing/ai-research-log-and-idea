"""Talk to GLT: ask it things, teach it a fact, take the fact away again.

Everything is plain English, but only the *labels* are: the model's vocabulary is 200 invented
symbols, relabelled by lexicon.py so the demo reads naturally. The model has no knowledge of any
real person -- it learned relations between arbitrary tokens, which is the point, since nothing
it says can have leaked in from pretraining.

  who is Anna's mentor?                ask -- any phrasing, and two hops work
  who is Anna's mentor's manager?      two hops: the workspace loop, not one forward pass
  Anna's mentor is Ben                 teach -- writes one fact into one slot
  forget Anna's mentor                 empty that slot, and drop what rested on it
  where is Anna's mentor from?         which slot, and memory or backbone?
  should I store Anna's mentor is Ben  the MDL admission score, in bits
  board                                every line on the workspace, and what produced it

Questions are not pattern-matched. They go through the three specialists of THEORY.md §8: the
parser turns any wording into the canonical token line (glt/language.py), the workspace loop
splits it into hops and records what produced what (glt/workspace.py), and the slots answer one
hop per forward pass. `teach`, `forget` and `bits` stay on fixed patterns because they are
commands rather than questions. Words the parser has never seen are dropped as noise.
"""

import re
import sys
from pathlib import Path

import torch

from glt.admission import surprise_bits
from glt.corpus import build, prefix
from glt.language import extract
from glt.language import load as load_parser
from glt.model import GPT, GPTConfig
from glt.universe import NONE, Fact, generate
from glt.workspace import follow, resolve, retract, solve, told, trace
from lexicon import Lexicon

torch.serialization.add_safe_globals([GPTConfig])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MEMORY = "out/memory.pt"   # slots taught interactively, kept across runs
#: The abstaining backbone: trained on questions with no answer, so silence is a token it can
#: emit. Costs 0.005 end-to-end accuracy against out/online.pt and removes confident
#: confabulation entirely (0.22 -> 0.00), which is the trade this project should always take.
MODEL = "out/online_abstain.pt"

W = r"([A-Za-z]+)"
#: (intent, pattern, order) -- "sr" captures subject then relation ("Anna's mentor"),
#: "rs" captures relation then subject ("the mentor of Anna"). Getting this backwards
#: silently answers about a different fact, so it is stated per pattern rather than sniffed.
PATTERNS = [
    ("teach", re.compile(rf"^(?:teach\s+)?{W}'s {W} is {W}\.?$", re.I), "sr"),
    ("teach", re.compile(rf"^(?:the\s+)?{W} of {W} is {W}\.?$", re.I), "rs"),
    ("forget", re.compile(rf"^forget\s+(?:about\s+)?{W}'s {W}\.?$", re.I), "sr"),
    ("where", re.compile(rf"^where.*?{W}'s {W}\??$", re.I), "sr"),
    ("bits", re.compile(rf"^(?:should i store|how surprising is)\s+{W}'s {W} is {W}\??$", re.I), "sr"),
]


def parse(line):
    """Returns (intent, [relation, subject, object?]) regardless of how it was phrased."""
    for intent, pattern, order in PATTERNS:
        match = pattern.match(line.strip())
        if match:
            groups = list(match.groups())
            if order == "sr":
                groups[0], groups[1] = groups[1], groups[0]
            return intent, groups
    return None, []


def understand(line, lex, tokenizer):
    """English -> the line the parser reads: real names swapped back for the model's own words,
    punctuation split off, and any word the parser never saw dropped as noise."""
    words = re.findall(r"[a-z]+|'s|[?.,:;]", line.lower())
    return " ".join(word for word in ((lex.to_raw(w) or w) for w in words) if word in tokenizer.stoi)


def load(path=MODEL):
    ck = torch.load(path, map_location=DEVICE, weights_only=True)
    model = GPT(ck["config"]).to(DEVICE)
    model.load_state_dict(ck["model"])
    return model.requires_grad_(False).eval()


def slots_of(model):
    return next(b.slots for b in model.transformer.h if b.slots is not None)


def load_memory(model, path=MEMORY):
    """Taught facts are just slot rows, so persistence is those rows and nothing else."""
    if not Path(path).exists():
        return {}
    learned = torch.load(path, map_location=DEVICE, weights_only=True)
    with torch.no_grad():
        for slot, row in learned.items():
            slots_of(model).values[slot] = row
    return learned


def save_memory(learned, path=MEMORY):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(learned, path)


def ids_for(corpus, rel, subj):
    return torch.tensor([corpus.tokenizer.encode(prefix(Fact(subj, (rel,), "?")))], device=DEVICE)


@torch.no_grad()
def address(model, corpus, rel, subj):
    return model.address(ids_for(corpus, rel, subj)).max().item()


@torch.no_grad()
def answer(model, corpus, rel, subj):
    probs = model(ids_for(corpus, rel, subj))[0][0, -1].float().softmax(-1)
    best = probs.argmax().item()
    return corpus.tokenizer.itos[best], probs[best].item()


def teach(model, corpus, rel, subj, obj, steps=300, lr=0.05):
    """Gradient on one slot row; every other parameter in the model is frozen."""
    slot = address(model, corpus, rel, subj)
    values = slots_of(model).values
    values.requires_grad_(True)
    mask = torch.zeros_like(values[:, :1])
    mask[slot] = 1.0
    handle = values.register_hook(lambda g: g * mask)

    gold = torch.tensor([corpus.tokenizer.stoi[obj]], device=DEVICE)
    optimizer = torch.optim.Adam([values], lr=lr)
    for step in range(steps):
        loss = torch.nn.functional.cross_entropy(model(ids_for(corpus, rel, subj))[0][:, -1].float(), gold)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if loss.item() < 0.01:
            break
    handle.remove()
    values.requires_grad_(False)
    return slot, step + 1


def truth_of(known, query):
    """Walk the ground-truth universe, so the demo can admit it is wrong rather than assert.

    Only `follow` has a ground truth cheap enough to check this way; the rest answer for
    themselves in operations.py.
    """
    if query.op != "follow":
        return None
    subject = query.subject
    for relation in query.path:
        subject = known.get((relation, subject))
        if subject is None:
            return None
    return subject


def show(lex, query):
    """A question in display names: Anna's mentor's manager."""
    if query.op == "follow":
        return "'s ".join(lex.to_nice(word) for word in (query.subject, *query.path))
    if query.op == "reverse":
        return f"everyone whose {lex.to_nice(query.path[0])} is {lex.to_nice(query.subject)}"
    return f"{query.op} of ({'; '.join(show(lex, inner) for inner in query.inner)})"


def render(lex, answer):
    """An answer in display names -- one name, a set of them, a number, or yes/no."""
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    if isinstance(answer, tuple):
        return ", ".join(lex.to_nice(word) for word in answer) or "nobody"
    return lex.to_nice(answer) if isinstance(answer, str) else str(answer)


def reply(lex, query, answer):
    """One line of English for an answer, shaped by the operation that produced it."""
    if query.op == "follow":
        return f"{show(lex, query)} is {render(lex, answer)}."
    if query.op == "reverse":
        return (f"{render(lex, answer)} -- that is everyone whose "
                f"{lex.to_nice(query.path[0])} is {lex.to_nice(query.subject)}.")
    if query.op == "count":
        inner = query.inner[0]
        return (f"{answer} -- that is how many have {lex.to_nice(inner.subject)} "
                f"as their {lex.to_nice(inner.path[0])}.")
    return "Yes, the same person." if answer else "No, two different people."


class Slots:
    """The memory specialist as the board sees it: one hop, or a batched scan for the reverse."""

    def __init__(self, model, corpus, entities):
        self.model, self.corpus, self.entities = model, corpus, entities

    def follow(self, relation, subject):
        """`NONE` is the model saying it holds no such fact, which the board reads as an open
        question rather than as an answer."""
        word, confidence = answer(self.model, self.corpus, relation, subject)
        return (None if word == NONE else word), confidence

    @torch.no_grad()
    def scan(self, relation, target, floor=0.99):
        """Everyone whose `relation` points at `target`, in one forward pass over all entities.

        No inverse index -- that would be an external fact table, which THEORY.md §8 rules out.
        The reverse question is answered by asking the *forward* one of everybody at once.

        `floor` is the price of asking a question of entities that have no answer to it: the
        argmax always names somebody, so an entity with no `relation` at all still votes. Held
        facts are answered at >= 0.998 (median 1.000) and confabulations at a median of 0.591,
        so the cut sits below every true edge rather than between them -- but it is a threshold,
        not a proof, and `operations.py` reports what still gets through.
        """
        ids = torch.tensor([self.corpus.tokenizer.encode(prefix(Fact(entity, (relation,), "?")))
                            for entity in self.entities], device=DEVICE)
        probs = self.model(ids)[0][:, -1].float().softmax(-1)
        wanted = self.corpus.tokenizer.stoi[target]
        return [(entity, probs[i, wanted].item()) for i, entity in enumerate(self.entities)
                if probs[i].argmax().item() == wanted and probs[i, wanted] >= floor]


def main(path=MODEL):
    universe = generate(0, entities=200, relations=20, out_degree=10)
    corpus, model = build(universe), load(path)
    lex = Lexicon(universe.entities, universe.relations)
    known = {(f.path[0], f.subject): f.object for f in universe.base}
    learned = load_memory(model)
    reader, tokenizer = load_parser(device=DEVICE)
    memory = Slots(model, corpus, universe.entities)
    board = []

    print(f"GLT -- {model.get_num_params()/1e6:.1f}M parameters, {slots_of(model).values.shape[0]} memory slots.")
    if learned:
        print(f"Recalled {len(learned)} fact(s) you taught me previously.")
    print("Ask me about people. I know some already; you can teach me more, and take them back.")
    print('Try:  who is Anna\'s mentor?   |   names   |   help   |   quit\n')

    while True:
        try:
            line = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not line:
            continue
        low = line.lower()
        if low in ("quit", "exit", "bye"):
            return
        if low == "help":
            print(__doc__.split("\n\n")[2])
            continue
        if low in ("names", "people"):
            print("glt > I know these people:", ", ".join(lex.to_nice(e) for e in sorted(universe.entities)[:16]), "...")
            continue
        if low == "board":
            print("glt > the workspace:" if board else "glt > the workspace is empty.")
            for note in board:
                rests = (f"   <- {' + '.join(show(lex, follow(f.subject, f.path)) for f in note.premises)}"
                         if note.premises else "")
                print(f"      [{note.source:>7}] {show(lex, note.query)}"
                      f" is {'?' if note.open else render(lex, note.answer)}{rests}")
            continue
        if low in ("relations", "kinds"):
            print("glt > I know these relationships:", ", ".join(lex.to_nice(r) for r in sorted(universe.relations)))
            continue
        if low.startswith("what do you know about"):
            who = lex.to_raw(low.replace("?", "").split()[-1])
            facts = [(r, o) for (r, s), o in known.items() if s == who]
            print("glt > " + ("; ".join(f"{lex.to_nice(r)} is {lex.to_nice(o)}" for r, o in facts[:6])
                              if facts else "I don't know anyone by that name."))
            continue

        intent, words = parse(line)
        if not intent:
            # Not a command, so it is a question: parser -> workspace -> slots, one hop at a time.
            query = extract(reader, tokenizer, [understand(line, lex, tokenizer)], DEVICE)[0]
            if query is None:
                print("glt > That is not a question I can express. I can follow a relation"
                      " (Anna's mentor), go backwards (who has Anna as their mentor), count,"
                      " and compare two of those.")
                continue
            board, note = solve(query, memory, board)
            if note is None or note.confidence < 0.5:
                print(f"glt > I don't really know {show(lex, query)}.")
                continue
            truth = truth_of(known, query)
            wrong = "" if truth in (None, note.answer) else f"  (actually {lex.to_nice(truth)}, so I am wrong)"
            print(f"glt > {reply(lex, query, note.answer)} ({note.confidence:.0%} sure){wrong}")
            hops = trace(board, note)
            if len(hops) > 1:
                for hop in hops:
                    print(f"      via {show(lex, hop.query)} is {render(lex, hop.answer)}")
            continue
        raw = [lex.to_raw(w) for w in words]
        if any(r is None for r in raw):
            missing = [w for w, r in zip(words, raw) if r is None]
            print(f"glt > I don't know the name {' or '.join(missing)}. Try `names` or `relations`.")
            continue
        rel, subj = raw[0], raw[1]

        if intent == "teach":
            before, sure = answer(model, corpus, rel, subj)
            if before != raw[2] and sure >= 0.5:
                print(f"glt > That contradicts me: I have {show(lex, follow(subj, (rel,)))}"
                      f" as {lex.to_nice(before)}. Yours is newer, so I'll take it.")
            board, dropped = resolve([*board, told(follow(subj, (rel,)), raw[2])])
            slot, steps = teach(model, corpus, rel, subj, raw[2])
            word, p = answer(model, corpus, rel, subj)
            print(f"glt > Got it. {show(lex, follow(subj, (rel,)))} is {lex.to_nice(word)}. ({p:.0%} sure)")
            learned[slot] = slots_of(model).values[slot].detach().clone()
            save_memory(learned)
            print(f"      Stored in slot {slot} after {steps} steps. I used to say {lex.to_nice(before)}.")
            print(f"      Written to {MEMORY}, so I will still know it next time you start me.")
            known[(rel, subj)] = raw[2]   # you just corrected me, so that is the truth now
            if dropped:
                print(f"      Dropped {len(dropped)} line(s) that rested on the old answer.")
        elif intent == "forget":
            slot = address(model, corpus, rel, subj)
            with torch.no_grad():
                slots_of(model).values[slot] = 0
            learned.pop(slot, None)
            save_memory(learned)
            word, p = answer(model, corpus, rel, subj)
            dropped = len(board) - len(retract(board, subj, (rel,)))
            board, _ = retract(board, subj, (rel,)), known.pop((rel, subj), None)
            print(f"glt > Forgotten. Slot {slot} is empty; now I'd only guess {lex.to_nice(word)} ({p:.0%}).")
            if dropped:
                print(f"      Dropped {dropped} line(s) from the workspace that rested on it.")
        elif intent == "where":
            slot = address(model, corpus, rel, subj)
            with_slot, p = answer(model, corpus, rel, subj)
            saved = slots_of(model).values[slot].clone()
            with torch.no_grad():
                slots_of(model).values[slot] = 0
            without, _ = answer(model, corpus, rel, subj)
            with torch.no_grad():
                slots_of(model).values[slot] = saved
            source = "that memory slot" if with_slot != without else "the network itself, not memory"
            print(f"glt > {show(lex, follow(subj, (rel,)))} lives in slot {slot}. I answer {lex.to_nice(with_slot)} ({p:.0%});")
            print(f"      with that slot blanked I would say {lex.to_nice(without)}, so the answer comes from {source}.")
        elif intent == "bits":
            bits = surprise_bits(model, corpus, [Fact(subj, (rel,), raw[2])], DEVICE).item()
            verdict = "worth storing" if bits >= 1 else "I already know this — not worth a slot"
            print(f"glt > That would cost me {bits:.1f} bits of surprise: {verdict}.")


if __name__ == "__main__":
    main(*sys.argv[1:])
