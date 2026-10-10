"""Invented mini-language world (the user's "let it learn a simple language", 2026-09-02):
a closed lexicon of invented words with generated facts — definitions and relations over
small closed sets. Every line is true by construction and appears in no corpus, so
acquisition-from-zero is measurable and the world itself is the grader."""
import torch

CONS, VOW = "bdfgklmprstvz", "aeiou"
KINDS = "bird fish animal tree flower stone tool game drink fruit".split()
COLORS = "red blue green black white yellow brown pink".split()
VERBS = ["eats", "likes", "fears", "makes"]


def _n(g, hi):
    return int(torch.randint(hi, (), generator=g))


def _word(g):
    return "".join(CONS[_n(g, 13)] + VOW[_n(g, 5)] for _ in range(2 + _n(g, 2)))


def make_world(n, seed):
    """n entities -> 2n true lines: 'a <word> is a <color> <kind>' + one relation each."""
    g = torch.Generator().manual_seed(seed)
    words = []
    while len(words) < n:
        w = _word(g)
        if w not in words:
            words.append(w)
    lines = []
    for w in words:
        lines.append(f"a {w} is a {COLORS[_n(g, len(COLORS))]} {KINDS[_n(g, len(KINDS))]}")
        lines.append(f"the {w} {VERBS[_n(g, len(VERBS))]} the {words[_n(g, n)]}")
    return lines
