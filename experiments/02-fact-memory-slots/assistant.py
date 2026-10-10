"""GLT as a product: talk normally, and the memory is exact.

Three specialists, as THEORY.md §8 describes them, but with the language one *bought* rather
than trained (§9.3):

    language    Gemma 4 E2B-it turns what you say into one line of a command language
    memory      the GLT backbone holds each fact in one addressed slot
    logic       the workspace chains, counts, compares, and records what rests on what

**The memory starts empty.** PROJECT_GOAL.md §17 measured that 99% of this model's factual
knowledge lives in its slots, so zeroing them leaves a backbone that still knows *how to read a
fact* and holds none -- a blank notebook and someone who can read. Everything it then knows, you
taught it.

**It cannot make a fact up.** Gemma only ever reads: it names the shape of what you said and
never writes an answer, and the reply you see is templated from what the memory returned. A name
Gemma imagines resolves to nothing, because only an explicit statement of fact may bind a new
name (glt/alias.py).

**What it knows survives being turned off.** `out/memory.pt` holds the slot rows -- the facts
themselves, as weights -- and `out/alias.json` holds which word means which symbol. Both are
written after every change. Derived conclusions are not saved and are recomputed on demand,
which is the §18 decision that keeps guesses out of the fact store.

    python assistant.py            # ~10 GB of VRAM for Gemma, 12 MB for the memory
    python assistant.py --forget-everything
"""

import sys

import torch

from glt.alias import ENTITY, RELATION, Alias
from glt.corpus import build
from glt.llm import Parser
from glt.universe import generate
from glt.workspace import count, follow, resolve, retract, reverse, same, solve, told, trace
from chat import DEVICE, Slots, address, load, load_memory, save_memory, slots_of, teach

GEMMA, CACHE = "google/gemma-4-E2B-it", "D:/hf-cache"
HELP = ("I can remember relationships between people and things. Try: \"my sister is Mia\","
        " \"who is my sister?\", \"who is my sister's boss?\", \"who has Mia as their sister?\","
        " \"how many people have Mia as their sister?\", \"forget my sister\"."
        " Ask me `what do you know` or `why` at any time.")


def whose(alias, symbol) -> str:
    """Possessive, in your words -- and you are `you`, not `me`."""
    word = alias.word(symbol)
    return "your" if word.lower() == "me" else f"{word}'s"


def plausible(word: str) -> bool:
    """Worth spending one of 200 permanent symbols on.

    A misheard statement is worse than a misheard question: the question is wrong once, the
    statement is wrong forever and has taken a slot with it. `m` and `123` are what a typo and
    a misparse look like.
    """
    return len(word.strip()) >= 2 and any(character.isalpha() for character in word)


def snapshot(alias, learned, board):
    """Everything one statement changes, so `undo` can put it all back."""
    return (dict(alias.symbols), dict(alias.words), dict(alias.kinds),
            {kind: list(pool) for kind, pool in alias.pool.items()}, dict(learned), list(board))


def named(alias, symbol) -> str:
    """One symbol in your words -- and you are `you`, not `me`."""
    word = alias.word(symbol)
    return "you" if word.lower() == "me" else word


def render(alias, answer) -> str:
    """An answer in your words -- one name, a set of them, a number, or yes/no."""
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    if isinstance(answer, tuple):
        return ", ".join(named(alias, symbol) for symbol in answer) or "nobody"
    return named(alias, answer) if isinstance(answer, str) else str(answer)


def as_query(alias, intent):
    """An Intent over English words -> a Query over symbols, or the word that was not known."""
    symbols = [alias.symbol(word) for word in intent.args]
    missing = next((word for word, symbol in zip(intent.args, symbols) if symbol is None), None)
    if missing:
        return None, missing
    if intent.op == "follow":
        return follow(symbols[0], tuple(symbols[1:])), None
    if intent.op == "reverse":
        return reverse(symbols[0], symbols[1]), None
    if intent.op == "count":
        return count(reverse(symbols[0], symbols[1])), None
    return same(follow(symbols[0], (symbols[1],)), follow(symbols[2], (symbols[3],))), None


def main(*flags) -> None:
    universe = generate(0, entities=200, relations=20, out_degree=10)
    corpus, model = build(universe), load()
    with torch.no_grad():
        slots_of(model).values.zero_()          # a blank notebook; the reader is intact
    alias = Alias(universe.entities, universe.relations)
    learned = {} if "--forget-everything" in flags else load_memory(model)
    if "--forget-everything" in flags:
        save_memory(learned)
        alias.symbols, alias.words, alias.kinds = {}, {}, {}
        alias.save()

    print(f"loading {GEMMA} ...", flush=True)
    language, memory, board = Parser(GEMMA, DEVICE, CACHE), Slots(model, corpus, universe.entities), []
    undo = None
    print(f"GLT. {len(learned)} fact(s) remembered from last time. {HELP}\n")

    while True:
        try:
            said = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        if not said:
            continue
        if said.lower() in ("quit", "exit", "bye"):
            return
        if said.lower() in ("help", "?"):
            print(f"glt > {HELP}")
            continue
        if said.lower() == "undo":
            if undo is None:
                print("glt > Nothing to undo.")
                continue
            (alias.symbols, alias.words, alias.kinds, alias.pool, learned, board), rows = undo, undo[4]
            with torch.no_grad():
                slots_of(model).values.zero_()
                for slot, row in rows.items():
                    slots_of(model).values[slot] = row
            save_memory(learned), alias.save()
            undo = None
            print("glt > Undone -- that never happened.")
            continue
        if said.lower().startswith("what do you know"):
            print(f"glt > people and things: {', '.join(alias.known(ENTITY)) or 'none yet'}")
            print(f"      relationships: {', '.join(alias.known(RELATION)) or 'none yet'}")
            continue
        if said.lower() in ("why", "why?", "show your work"):
            print("glt > " + ("\n      ".join(f"{alias.word(n.query.subject)}'s "
                                              f"{alias.word(n.query.path[0])} is {render(alias, n.answer)}"
                                              for n in board if not n.open and n.query.op == "follow")
                             or "nothing on the board yet."))
            continue

        intent = language.read([said])[0]
        if intent.op == "none":
            print("glt > I only deal in facts about relationships -- try \"my sister is Mia\","
                  " or ask me one. Say `help` for the rest.")
            continue

        if intent.op == "teach":
            subject, relation, obj = intent.args
            junk = [word for word in intent.args if not plausible(word)]
            if junk:
                print(f"glt > \"{junk[0]}\" doesn't look like a name to me, and I'd be stuck with"
                      f" it. Say it again with the full word?")
                continue
            undo = snapshot(alias, learned, board)
            symbols = [alias.bind(subject, ENTITY), alias.bind(relation, RELATION),
                       alias.bind(obj, ENTITY)]
            if None in symbols:
                print("glt > I've run out of room for new names.")
                continue
            before, sure = memory.follow(symbols[1], symbols[0])
            if before is not None and before != symbols[2] and sure >= 0.5:
                print(f"glt > That contradicts what I had: {whose(alias, symbols[0])} {relation} was "
                      f"{alias.word(before)}. Yours is newer, so I'll take it.")
            board, dropped = resolve([*board, told(follow(symbols[0], (symbols[1],)), symbols[2])])
            slot, steps = teach(model, corpus, symbols[1], symbols[0], symbols[2])
            learned[slot] = slots_of(model).values[slot].detach().clone()
            save_memory(learned)
            alias.save()
            print(f"glt > Got it: {whose(alias, symbols[0])} {relation} is {obj}."
                  + (f" Dropped {len(dropped)} thing(s) that rested on the old answer." if dropped else "")
                  + ("  (say `undo` if I misheard)" if undo else ""))
            continue

        if intent.op == "forget":
            subject, relation = intent.args
            symbols = [alias.symbol(subject), alias.symbol(relation)]
            if None in symbols:
                print(f"glt > I never knew anything about {subject} 's {relation}.")
                continue
            undo = snapshot(alias, learned, board)
            slot = address(model, corpus, symbols[1], symbols[0])
            with torch.no_grad():
                slots_of(model).values[slot] = 0
            learned.pop(slot, None)
            save_memory(learned)
            dropped = len(board) - len(retract(board, symbols[0], (symbols[1],)))
            board = retract(board, symbols[0], (symbols[1],))
            print(f"glt > Forgotten: {whose(alias, symbols[0])} {relation}."
                  + (f" {dropped} thing(s) that rested on it went too." if dropped else ""))
            continue

        query, missing = as_query(alias, intent)
        if missing:
            print(f"glt > I don't know anything called \"{missing}\" yet. Tell me and I'll remember.")
            continue
        board, note = solve(query, memory, board)
        if note is None or note.confidence < 0.5:
            print("glt > I don't know that one.")
            continue
        answer = render(alias, note.answer)
        hops = trace(board, note)
        print(f"glt > {answer}" + (f"  ({note.confidence:.0%} sure)" if note.confidence < 0.99 else ""))
        for hop in hops if len(hops) > 1 else []:
            print(f"      via {whose(alias, hop.query.subject)} {alias.word(hop.query.path[0])}"
                  f" is {render(alias, hop.answer)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
