"""Base-curriculum stage streams 2-6 (design-interface 7.7; C34 build protocol). Every line is
generated with its answer computed by construction — graded text, zero external labels.
Stage 2 qa: being addressed (echo / arithmetic / comparison / counting).
Stage 3 cot: worked steps, every intermediate verified (T: line before A:).
Stage 4 tools: run:/read: transcripts with REAL outputs (calc for code, corpus lines for read).
Stage 5 idk: invented-entity questions where "I don't know" IS the correct answer.
Stage 6 quiz: cloze self-quiz over the child's own corpus (span-verifiable)."""
from pathlib import Path

import torch

from cl.ladder import calc, expr

WORDS = ("cat dog ball tree house water bird fish milk bread hand door book star rain "
         "sun moon car boat song").split()


def _n(gen, hi=10):
    return int(torch.randint(hi, (), generator=gen))


def _w(gen):
    return WORDS[_n(gen, len(WORDS))]


def qa(gen):
    k = _n(gen, 4)
    if k == 0:
        w = f"the {_w(gen)} and the {_w(gen)}"
        return f"Q: say {w}\nA: {w}\n"
    if k == 1:
        e = expr(1 + _n(gen, 2), gen)
        return f"Q: what is {e}\nA: {calc(e)}\n"
    if k == 2:
        x, y = _n(gen), _n(gen)
        return f"Q: which is bigger {x} or {y}\nA: {max(x, y)}\n" if x != y else f"Q: which is bigger {x} or {y}\nA: same\n"
    w, c = _w(gen), 2 + _n(gen, 4)
    return f"Q: count the {w}s: {' '.join([w] * c)}\nA: {c}\n"


def _steps(e):
    """Innermost-first evaluation trace of a fully parenthesised expression."""
    out = []
    while "(" in e:
        i = e.rfind("(")
        j = e.index(")", i)
        val = calc(e[i:j + 1])
        out.append(f"{e[i + 1:j]}={val}")
        e = e[:i] + str(val) + e[j + 1:]
    return out


def cot(gen):
    e = expr(2 + _n(gen, 2), gen)
    return f"Q: what is {e}\nT: {'. '.join(_steps(e))}.\nA: {calc(e)}\n"


def tools(gen, lines):
    if _n(gen, 2):
        e = expr(1 + _n(gen, 2), gen)
        return f"run: print({e})\nout: {calc(e)}\n"
    ln = lines[_n(gen, len(lines))]
    return f"read: {ln.split()[0]}\nout: {ln}\n"


def idk(gen):
    if _n(gen, 2):
        fake = "".join("bdfgklmprstvz"[_n(gen, 13)] for _ in range(6))
        return f"Q: what is a {fake}\nA: I don't know\n"
    e = expr(1 + _n(gen, 2), gen)
    return f"Q: what is {e}\nA: {calc(e)}\n"


def seek(gen, lines):
    """C39 format-mini: the reach chain (Q -> search -> out -> A), answers by construction."""
    ln = lines[_n(gen, len(lines))].split()
    i = 1 + _n(gen, len(ln) - 1)
    blank = " ".join("___" if j == i else w for j, w in enumerate(ln))
    kw = " ".join(w for j, w in enumerate(ln) if j != i and len(w) > 3)[:40]
    return f"Q: fill the blank: {blank}\nsearch: {kw}\nout: {' '.join(ln)}\nA: {ln[i]}\n"


def wcot(gen):
    """v3 chain-of-thought stage (user directive 2026-09-02): 2-hop derivations over invented
    facts — bind a fact from context, derive, answer; every step true by construction. Fresh
    invented words per unit so the FORMAT generalizes instead of the content memorizing."""
    from cl import world as W
    w = W._word(gen)
    color = W.COLORS[_n(gen, len(W.COLORS))]
    kind = W.KINDS[_n(gen, len(W.KINDS))]
    fact = f"a {w} is a {color} {kind}"
    k = _n(gen, 3)
    if k == 0:
        c2 = W.COLORS[_n(gen, len(W.COLORS))]
        yes = c2 == color
        return (f"text: {fact}\nQ: is a {w} {c2}\n"
                f"T: a {w} is {color}. {color} {'is' if yes else 'is not'} {c2}.\n"
                f"A: {'yes' if yes else 'no'}\n")
    if k == 1:
        k2 = W.KINDS[_n(gen, len(W.KINDS))]
        yes = k2 == kind
        return (f"text: {fact}\nQ: is a {w} a {k2}\n"
                f"T: a {w} is a {kind}.\nA: {'yes' if yes else 'no'}\n")
    return f"text: {fact}\nQ: what color is a {w}\nT: a {w} is a {color} {kind}.\nA: {color}\n"


def quiz(gen, lines):
    ln = lines[_n(gen, len(lines))].split()
    i = 1 + _n(gen, len(ln) - 1)
    blank = " ".join("___" if j == i else w for j, w in enumerate(ln))
    return f"text: {' '.join(ln)}\nQ: fill the blank: {blank}\nA: {ln[i]}\n"


def _corpus_lines(track="100M", min_words=5, max_words=12, n=20000):
    src = Path(__file__).parents[1] / "data" / "babylm" / track / "aochildes.txt"
    lines = [ln.strip() for ln in src.read_text(encoding="utf-8", errors="ignore").splitlines()]
    return [ln for ln in lines if min_words <= len(ln.split()) <= max_words][:n]


def stream(stage, n_tokens, seed, track="100M"):
    """Stage bytes: ~n_tokens of generated, graded lines."""
    gen = torch.Generator().manual_seed(seed)
    lines = _corpus_lines(track) if stage in ("tools", "quiz", "seek") else None
    fn = {"qa": qa, "cot": cot, "idk": idk, "wcot": wcot, "tools": lambda g: tools(g, lines),
          "quiz": lambda g: quiz(g, lines), "seek": lambda g: seek(g, lines)}[stage]
    buf = []
    size = 0
    while size < n_tokens:
        s = fn(gen)
        buf.append(s)
        size += len(s)
    return "".join(buf).encode()
