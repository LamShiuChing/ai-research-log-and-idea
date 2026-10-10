"""Generation for the Stage-C flywheel (design-interface §7.6-7.7; CLAIMS C35). Batched ragged
sampling from the tier-2 GPT with a REAL `run:` tool: the moment a row completes a
`run: print(E)` line, the interpreter's actual output is injected into that row's context and
generation continues from it — the first closed loop (§7.1 condition 3). Rows are byte-lists of
different lengths; right-padded batching is exact under causal attention because pad positions
sit strictly to the right of every position we read. The interpreter (cl.ladder.calc,
AST-restricted) is the only judge of the child's units — §5's rule."""
import re

import torch
import torch.nn.functional as F

from cl.ladder import calc, expr
from cl.tools import TOOL_LINE

TOOL = re.compile(r"run: print\((.*)\)\s*$")
SEARCH = re.compile(r"search: (.+)$")
UNIT = re.compile(r"Q: what is ([^\n]+)\n(?:(?:T|run|out): [^\n]*\n)*A: (-?\d+)\n")
QUIZ = re.compile(r"Q: fill the blank: ([^\n]+)\nA: ([^\n]+)\n")


def depth(expr_src):
    """Max paren nesting — the skill-unit index of an expression (C36 targeting)."""
    d = m = 0
    for c in expr_src:
        d += (c == "(") - (c == ")")
        m = max(m, d)
    return m


def lookup(query, lines):
    """The library's search tool: the line sharing the most words with the query (C38)."""
    q = set(query.split())
    return max(lines, key=lambda ln: len(q & set(ln.split())))


def _last_line(row):
    """Text of the line a trailing newline just completed."""
    s = bytes(row[:-1])
    return s[s.rfind(b"\n") + 1:].decode(errors="ignore")


@torch.no_grad()
def sample(net, rows, n_new, gen, temp=0.8, top_k=40, tool=False, library=None, tok=None, tools=None):
    """Extend each byte-list in rows by n_new tokens (temp=0 is greedy). tool=True wires the
    live run: interceptor, library wires search:; injected out: bytes lengthen a row beyond
    its sampled count. tok + tools (cl/tools.py) is the same closed loop on BPE rows: a completed
    `name: arg` line whose name is in `tools` gets its real output injected as `out:`."""
    dev = next(net.parameters()).device
    for _ in range(n_new):
        ctx = [r[-net.block:] for r in rows]
        lens = [len(c) for c in ctx]
        x = torch.zeros(len(rows), max(lens), dtype=torch.long, device=dev)
        for i, c in enumerate(ctx):
            x[i, :len(c)] = torch.tensor(c, device=dev)
        with torch.autocast("cuda", torch.bfloat16):
            logits = net(x)[0][torch.arange(len(rows), device=dev), torch.tensor(lens, device=dev) - 1]
        if temp == 0:
            nxt = logits.argmax(-1).tolist()
        else:
            v, ix = logits.float().topk(top_k)
            j = torch.multinomial(F.softmax(v.cpu() / temp, -1), 1, generator=gen).squeeze(1)
            nxt = ix.cpu()[torch.arange(len(rows)), j].tolist()
        for i, t in enumerate(nxt):
            rows[i].append(t)
            if tools is not None and "\n" in tok.decode([t]):
                text = tok.decode(rows[i][-128:])
                m = TOOL_LINE.match(text[:text.rfind("\n")].rsplit("\n", 1)[-1])  # the line this token closed
                if m and m.group(1) in tools:
                    rows[i].extend(tok.encode(f"out: {tools[m.group(1)](m.group(2))}\n").ids)
                continue
            if t == 10 and (tool or library is not None):  # byte-mode newline; BPE rows must keep tools off
                line = _last_line(rows[i])
                m = tool and TOOL.match(line)
                if m:
                    try:
                        rows[i].extend(f"out: {calc(m.group(1))}\n".encode())
                    except Exception:
                        pass  # the model wrote an unrunnable call; the line stands ungraded
                m = library is not None and SEARCH.match(line)
                if m:
                    rows[i].extend(f"out: {lookup(m.group(1), library)}\n".encode())
    return rows


def grade(text):
    """Parse Q/T/A units from generated text and grade each with the interpreter.
    Yields (unit_text, correct, used_tool); malformed expressions are skipped."""
    for m in UNIT.finditer(text):
        try:
            ok = calc(m.group(1)) == int(m.group(2))
        except Exception:
            continue
        yield m.group(0), ok, "run:" in m.group(0)


def cloze_acc(net, quizzes, library=None, free=False):
    """Exact-match on cloze quizzes [(blanked_line, answer)]. Direct mode forces an immediate
    A:; free mode lets the model search first and also returns reach flags (C39's statistic)."""
    tail = "\n" if free else "\nA: "
    rows = [list(f"Q: fill the blank: {b}{tail}".encode()) for b, _ in quizzes]
    pl = [len(r) for r in rows]
    sample(net, rows, 40 if free else 12, None, temp=0, library=library)
    ok, reach = 0, []
    for (b, ans), r, p in zip(quizzes, rows, pl):
        text = bytes(r[p:]).decode(errors="ignore")
        m = re.search(r"A: ([^\n]*)" if free else r"^([^\n]*)", text)
        ok += (m.group(1) if m else "").strip() == ans
        reach.append("search:" in text)
    return ok / len(quizzes), reach


def questions(depth, n, seed):
    g = torch.Generator().manual_seed(seed)
    return [expr(depth, g) for _ in range(n)]


def accuracy(net, exprs, tool=False):
    """Greedy exact-match on held-out questions. tool=True puts a REAL run:/out: transcript in
    context (the tool ran; the model must ground A: in its output)."""
    rows = [list((f"Q: what is {e}\nrun: print({e})\nout: {calc(e)}\nA: " if tool
                  else f"Q: what is {e}\nA: ").encode()) for e in exprs]
    pl = [len(r) for r in rows]
    sample(net, rows, 8, None, temp=0)
    ok = 0
    for e, r, p in zip(exprs, rows, pl):
        try:
            ok += int(bytes(r[p:]).decode(errors="ignore").split("\n")[0]) == calc(e)
        except ValueError:
            pass
    return ok / len(exprs)
