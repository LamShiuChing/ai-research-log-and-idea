"""Serve the GPT-2 Mind. Run from lab/gpt:  python ..\\..\\unified\\serve_gpt2.py
Then open http://localhost:8343
"""
import json
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mind_gpt2 import MindGPT2

mind = MindGPT2(model_type=os.environ.get("MIND_CORE", "gpt2-xl"))

EXTRACT = [
    (re.compile(r"my name is (\w+)", re.I), "What is my name",
     "Nice to meet you, {0} — I'll remember that."),
    (re.compile(r"my favou?rite (\w+) is ([\w ]+)", re.I), "What is my favorite {0}",
     "Remembered: your favorite {0} is {1}."),
    (re.compile(r"i (?:like|love) playing ([\w ]+)", re.I), "What is my favorite game",
     "Noted — you like playing {0}. I'll remember it as your favorite game."),
    (re.compile(r"i (?:like|love) ([\w ]+)", re.I), "What do I like",
     "Noted — you like {0}. I'll remember that."),
    (re.compile(r"remember (?:that )?(.+?) is ([\w ]+)", re.I), "What is {0}",
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
        with open(os.path.join(HERE, "index_gpt2.html"), "rb") as f:
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
                    text = reply.format(*m.groups())
                    mind.note(q, text)
                    mind.buffer("G3u", question.format(*m.groups()), m.groups()[-1])
                    self._json({"answer": text, "organ": "memory (extracted + saved)"})
                    return
            self._json(mind.answer(q))
        elif self.path == "/api/remember":
            mind.remember(data["question"], data["answer"])
            self._json({"ok": True, "memories": len(mind.memories)})


print("gpt2 mind ready on http://localhost:8343")
HTTPServer(("127.0.0.1", 8343), Handler).serve_forever()
