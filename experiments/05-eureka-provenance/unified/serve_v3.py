"""Serve Mind v3 (SmolLM3 core, tool-driven).  python unified\\serve_v3.py  ->  http://localhost:8346"""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mind_v3 import MindV3

arg = sys.argv[1] if len(sys.argv) > 1 else None  # python serve_v3.py [out-v3-lora | ollama:qwen2.5:14b]
if arg and arg.startswith("ollama:"):
    from mind_local import MindLocal
    mind = MindLocal(arg.removeprefix("ollama:"))
else:
    mind = MindV3(adapter=arg)


def log_turn(q, answer, organ):
    with open(os.path.join(HERE, "chat_log.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.strftime("%H:%M:%S"), "q": q,
                            "a": answer, "organ": organ}, ensure_ascii=False) + "\n")


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
        if self.path == "/api/memory":
            self._json(mind.store.facts(list(mind.store)))
            return
        with open(os.path.join(HERE, "index_v3.html"), "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/api/ask":
            r = mind.answer(data["q"])
            log_turn(data["q"], r["answer"], r["organ"])
            self._json(r)
        elif self.path == "/api/remember":
            mind.store.remember(data["question"], data["answer"], data.get("source", "user"))
            self._json({"ok": True, "memories": len(mind.store)})


print("mind v3 (SmolLM3, tool-driven) ready on http://localhost:8346")
HTTPServer(("127.0.0.1", 8346), Handler).serve_forever()
