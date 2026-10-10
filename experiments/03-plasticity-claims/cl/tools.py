"""Live tools for the live loop (design-interface §7.13). A tool is a line the child writes —
`run: <code>`, `read: <path>`, `search: <query>`, `ask: <question>` — whose REAL output is
injected as the next `out:` line while generation continues (cl/gen.py's closed loop). Every
tool is a reality that grades: the interpreter, the file, the corpus, the human. `run:` executes
model-written code in a subprocess under a timeout inside SANDBOX — it is not a security boundary;
keep the loop on a machine you would let a child type on."""
import re
import subprocess
import sys
from pathlib import Path

TOOL_LINE = re.compile(r"^(run|read|search|ask): (.+)$")
SANDBOX = Path("sandbox")
OUT = 1500  # chars of tool output the child gets to read


def run(code, lang="python", timeout=5):
    SANDBOX.mkdir(exist_ok=True)
    cmd = [sys.executable, "-c", code] if lang == "python" else ["node", "-e", code]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=SANDBOX)
    except subprocess.TimeoutExpired:
        return "timeout"
    return (p.stdout + p.stderr).strip()[:OUT] or "(no output)"


def reader(root):
    """`read: <path>` within `root`; a path outside it is refused."""
    root = Path(root).resolve()

    def read(path):
        p = (root / path).resolve()
        if not p.is_relative_to(root) or not p.is_file():
            return f"no such file: {path}"
        return p.read_text(encoding="utf-8", errors="ignore")[:OUT]
    return read


def searcher(root, exts=(".txt", ".md")):
    """`search: <query>` over the paragraphs of every text file under `root`: the paragraph
    sharing the most words with the query (cl/gen.py's library lookup, over a folder)."""
    paras = [(p.name, para) for p in Path(root).rglob("*") if p.suffix in exts
             for para in p.read_text(encoding="utf-8", errors="ignore").split("\n\n") if para.strip()]

    def search(query):
        if not paras:
            return "no documents"
        q = set(query.lower().split())
        name, para = max(paras, key=lambda np: len(q & set(np[1].lower().split())))
        return f"[{name}] {' '.join(para.split())[:OUT]}"
    return search


def make(root, ask, lang="python"):
    """The tool table for one environment: docs folder `root`, `ask(question) -> answer`."""
    return dict(run=lambda code: run(code, lang), read=reader(root), search=searcher(root), ask=ask)
