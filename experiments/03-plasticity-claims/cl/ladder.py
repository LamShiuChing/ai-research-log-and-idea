"""Skill-ladder exercise streams (design-interface.md §7.3). Each line is an exercise whose
answer is computed by the interpreter at generation time — the external verifier IS the teacher
(§5's rule), so every byte of the stream is ground truth by construction.
Rung 1: arithmetic expressions, difficulty = nesting depth. Rung 2: one let-binding on top.
Rendered as bytes for the tier-2 instruments (a rung is a domain; C(t) per rung is ACC-a's
statistic). Evaluation is AST-restricted (ints, + - *, bound names only) — verify() grades
untrusted model output, so no eval()."""
import ast
import operator

import torch

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}


def calc(src, env=None):
    """Evaluate an expression allowing only int literals, + - *, and names bound in env."""
    def walk(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.Name) and node.id in (env or {}):
            return env[node.id]
        raise ValueError(ast.dump(node))
    return walk(ast.parse(src, mode="eval").body)


def expr(depth, gen):
    if depth == 0:
        return str(int(torch.randint(10, (), generator=gen)))
    op = "+-*"[int(torch.randint(3, (), generator=gen))]
    return f"({expr(depth - 1, gen)}{op}{expr(depth - 1, gen)})"


def rung1(depth, gen):
    e = expr(depth, gen)
    return f"{e}={calc(e)}\n"


def rung2(depth, gen):
    var = "xyz"[int(torch.randint(3, (), generator=gen))]
    val = int(torch.randint(10, (), generator=gen))
    body = expr(depth - 1, gen) if depth > 1 else str(int(torch.randint(10, (), generator=gen)))
    op = "+-*"[int(torch.randint(3, (), generator=gen))]
    e = f"({var}{op}{body})"
    return f"let {var}={val};{e}={calc(e, {var: val})}\n"


RUNGS = {1: rung1, 2: rung2}


def stream(rung, depth, n_lines, seed):
    """The exercise stream as bytes: one verified line per exercise."""
    gen = torch.Generator().manual_seed(seed)
    return "".join(RUNGS[rung](depth, gen) for _ in range(n_lines)).encode()


def verify(text):
    """Grade a stream (or model output) line by line: re-check every '=' claim. Returns (ok, n)."""
    ok = n = 0
    for line in text.decode(errors="ignore").splitlines():
        lhs, _, rhs = line.rpartition("=")
        env = {}
        if lhs.startswith("let "):
            bind, _, lhs = lhs.partition(";")
            var, val = bind[4:].split("=")
            env[var] = int(val)
        n += 1
        try:
            ok += calc(lhs, env) == int(rhs)
        except (SyntaxError, ValueError):
            pass
    return ok, n
