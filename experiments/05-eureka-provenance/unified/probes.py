"""The probe suite: python unified\probes.py [port]  (server must be running with a fresh 3-fact memory).
Seeds 5 user facts, 3 anchored source claims, 1 contested claim, then 38 questions across memory,
extraction, math, forget, world knowledge, skills. Prints one JSON line per probe, then the store."""
import json, sys, time, urllib.request, urllib.error

URL = f"http://127.0.0.1:{sys.argv[1] if len(sys.argv) > 1 else 8346}"


def post(path, obj):
    req = urllib.request.Request(URL + path, json.dumps(obj).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=300))


for _ in range(120):
    try:
        urllib.request.urlopen(URL, timeout=5); break
    except (urllib.error.URLError, ConnectionError):
        time.sleep(5)

for q, a in [("What is my favorite drink", "apple juice"), ("What is my favorite game", "minecraft"),
             ("What is my dog's name", "Mochi")]:
    post("/api/remember", {"question": q, "answer": a})
for q, good in [("What is my favorite color", "teal"), ("What is our project", "eureka"), ("What is my dog's name", "Mochi")]:
    post("/api/remember", {"question": q, "answer": good, "source": "wiki"})
    post("/api/remember", {"question": q, "answer": "banana", "source": "spam"})
    post("/api/remember", {"question": q, "answer": "banana", "source": "spam2"})
post("/api/remember", {"question": "What is my hometown", "answer": "Hong Kong", "source": "wiki"})
post("/api/remember", {"question": "What is my hometown", "answer": "Tokyo", "source": "spam"})
post("/api/remember", {"question": "What is my hometown", "answer": "Tokyo", "source": "spam2"})

PROBES = [
    ("MEMORY", "what's my name?"),
    ("MEMORY", "which colour do I like best?"),
    ("MEMORY", "what am I working on these days?"),
    ("MEMORY", "what's my pet called?"),
    ("MEMORY", "where am I from?"),
    ("MEMORY-UNKNOWN", "what's my favorite food?"),
    ("MEMORY-UNKNOWN", "how old am I?"),
    ("EXTRACT", "my name is not important"),
    ("EXTRACT", "I like the way you answer"),
    ("EXTRACT", "tell me something short about basketball"),
    ("EXTRACT", "my sister is called Anna"),
    ("EXTRACT", "my favoirte food is apple"),
    ("EXTRACT", "what's my loved sport?"),
    ("EXTRACT", "basketball"),
    ("EXTRACT", "what's my favorite sport?"),
    ("EXTRACT", "what's my sister's name?"),
    ("EXTRACT", "what's my favorite food?"),
    ("MATH", "what is 47 times 8"),
    ("MATH", "what is 29995*1823"),
    ("MATH", "what is 7 times 8 divided by 2"),
    ("MATH", "what is 100 / 4"),
    ("MATH", "what is 2 to the power of 10"),
    ("MATH-CHAIN", "what is 17 + 25 - 3"),
    ("MATH-CHAIN", "what is 12*12 + 47*8"),
    ("MATH-CHAIN", "I have 23 boxes with 17 items each. I give away 95 items. How many items are left?"),
    ("MATH-CHAIN", "A train travels 60 km per hour for 3 hours, then 45 km per hour for 2 hours. Total distance?"),
    ("FORGET", "forget my favorite sport"),
    ("FORGET", "what's my favorite sport?"),
    ("WORLD", "what is the capital of Australia?"),
    ("WORLD", "who wrote Hamlet?"),
    ("WORLD", "what year is it?"),
    ("WORLD", "who won the 2030 world cup?"),
    ("SKILL", "write a haiku about rain"),
    ("SKILL", "translate 'good morning, how are you' to French"),
    ("SKILL", "write a python function that reverses a string"),
    ("SKILL", "summarize in one sentence: The mitochondria is the powerhouse of the cell. It produces ATP through cellular respiration, which involves the citric acid cycle and oxidative phosphorylation."),
    ("FINAL", "what's my name?"),
    ("FINAL", "what's my favorite color?"),
]
for cat, q in PROBES:
    t = time.time()
    r = post("/api/ask", {"q": q})
    print(json.dumps({"cat": cat, "q": q, "a": r["answer"], "organ": r["organ"], "s": round(time.time() - t, 1)}, ensure_ascii=False))

print("=== /api/memory ===")
for row in json.load(urllib.request.urlopen(URL + "/api/memory")):
    print(row)
