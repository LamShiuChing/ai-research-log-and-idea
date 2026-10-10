"""Serve the unified Mind. Run from lab/gpt:  python ..\\..\\unified\\serve.py
Then open http://localhost:8342
"""
import json
import os
import pickle
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mind import Mind

# entities.pkl is generated locally by this repo — trusted input
with open(os.path.join(HERE, "..", "lab", "e7-skills", "entities.pkl"), "rb") as f:
    FACTS = pickle.load(f)["facts"]

seed = {f"What is the color of {n}": c for n, c in FACTS.items()}
mind = Mind("out-e9-base", seed)

EXTRACT = [  # (pattern, question, reply) — statements worth remembering, from plain chat
    (re.compile(r"my name is (\w+)", re.I), "What is my name",
     "Nice to meet you, {0} — I'll remember that."),
    (re.compile(r"remember (?:that )?(.+?) is (\w+)", re.I), "What is {0}",
     "Remembered: {0} is {1}."),
]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        with open(os.path.join(HERE, "index.html"), "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/ask":
            q = data["q"]
            for pat, question, reply in EXTRACT:
                m = pat.search(q)
                if m:
                    mind.remember(question.format(*m.groups()), m.groups()[-1])
                    self._json({"answer": reply.format(*m.groups()),
                                "organ": "memory (extracted + saved)"})
                    return
            self._json(mind.answer(q))
        elif self.path == "/api/remember":
            mind.remember(data["question"], data["answer"])
            self._json({"ok": True, "memories": len(mind.memories)})


print("mind ready on http://localhost:8342")
HTTPServer(("127.0.0.1", 8342), Handler).serve_forever()
