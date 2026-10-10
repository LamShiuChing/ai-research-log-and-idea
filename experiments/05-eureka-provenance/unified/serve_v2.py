"""Serve Mind v2 (SmolLM3 core).  python unified\\serve_v2.py  ->  http://localhost:8344"""
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from extract import FIRST_PERSON, explicit_fact, may_state_fact
from mind_v2 import MindV2
from store import GRADE, LABEL

mind = MindV2()


def log_turn(q, answer, organ):
    """Every conversation turn is readable afterwards — chat_log.jsonl."""
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
        with open(os.path.join(HERE, "index_v2.html"), "rb") as f:
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
            fm = re.match(r"forget (?:about )?(.+)", q.strip(), re.I)
            if fm:
                gone = mind.forget(fm.group(1))
                text = (f"Forgotten: {'; '.join(gone)}." if gone
                        else "I couldn't find a memory matching that.")
                mind.note(q, text)
                log_turn(q, text, "memory (forget)")
                self._json({"answer": text, "organ": "memory (forget)"})
                return
            ef = explicit_fact(q)
            if ef:
                question, answer, text = ef
                mind.store.remember(question, answer)
                mind.note(q, text)
                mind.buffer("G3u", question, answer)
                log_turn(q, text, "memory (extracted + saved)")
                self._json({"answer": text, "organ": "memory (extracted + saved)"})
                return
            prev = mind.history[-1]["content"] if mind.history else ""
            fact = mind.extract_fact(q) if may_state_fact(q, prev) else None
            r = mind.answer(q)
            r["facts"] = mind.last_facts
            if fact:  # stated in the user's own first-person words -> user-taught; a short context reply -> a guess
                source = "user" if FIRST_PERSON.search(q) else "chat"
                mind.store.remember(*fact, source=source)
                mind.buffer(GRADE[source], *fact)
                r["organ"] += f" · saved [{LABEL[GRADE[source]]}]: {fact[0]} = {fact[1]}"
            log_turn(q, r["answer"], r["organ"])
            self._json(r)
        elif self.path == "/api/remember":
            mind.store.remember(data["question"], data["answer"], data.get("source", "user"))
            self._json({"ok": True, "memories": len(mind.store)})


print("mind v2 (SmolLM3) ready on http://localhost:8344")
HTTPServer(("127.0.0.1", 8344), Handler).serve_forever()
