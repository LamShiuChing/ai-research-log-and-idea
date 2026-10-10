"""Synthesize verified tool-use traces to train Mind v3's core.  python unified\\traces.py

Every calc trace is executed by the calculator (oracle-verified); every remember trace
stores a value copied from the user's words (evidence-verified); recall and forget traces
answer from a store slice. Plain turns (no tool) use the untouched core's own reply as the
target, so its general ability is preserved rather than replaced. Conversations of 1-3
turns. Writes traces.jsonl: {"messages": [...]} without the system message.
"""
import json
import os
import random
import re

import torch

from mind_v3 import MindV3, calc

random.seed(7)
HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["Eric", "Anna", "Mochi", "Lena", "Omar", "Priya", "Tom", "Yuki", "Sam", "Nora", "Kai", "Ivy"]
VALUES = {"color": ["teal", "red", "navy", "green", "purple"], "food": ["apple", "sushi", "pizza", "noodles"],
          "drink": ["apple juice", "coffee", "green tea", "water"], "game": ["minecraft", "chess", "tetris"],
          "sport": ["basketball", "tennis", "swimming", "football"], "movie": ["Inception", "Amelie", "Heat"]}
CITIES = ["Hong Kong", "Tokyo", "Lisbon", "Toronto", "Nairobi", "Oslo"]
PLAIN = ["write a haiku about rain", "translate 'good morning, how are you' to French",
         "write a python function that reverses a string", "what is the capital of Australia?",
         "who wrote Hamlet?", "what year is it?", "tell me something short about basketball",
         "I like the way you answer", "hello", "thanks!", "explain what a mitochondrion does in one sentence",
         "give me three names for a cat", "what's the difference between a list and a tuple in python?",
         "summarize: The mitochondria is the powerhouse of the cell. It produces ATP through cellular respiration.",
         "write a limerick about a cat", "how do I boil an egg?", "what is machine learning?", "tell me a joke"]


def call(name, **args):
    return {"role": "assistant", "content": f'<tool_call>\n{json.dumps({"name": name, "arguments": args})}\n</tool_call>'}


def tool(name, result):
    return {"role": "tool", "content": f"{name} result: {result}"}


def say(text):
    return {"role": "assistant", "content": text}


def arithmetic():
    a, b, c, d = (random.randint(2, 99) for _ in range(4))
    big = random.randint(1000, 99999), random.randint(100, 9999)
    q, expr = random.choice([
        (f"what is {a} times {b}", f"{a}*{b}"), (f"{a} * {b}", f"{a}*{b}"), (f"what is {big[0]}*{big[1]}", f"{big[0]}*{big[1]}"),
        (f"calculate {a} + {b}", f"{a}+{b}"), (f"how much is {a} minus {b}?", f"{a}-{b}"), (f"what is {a} / {b}", f"{a}/{b}"),
        (f"what is {a} to the power of 3", f"{a}**3"), (f"what is {a} + {b} - {c}", f"{a}+{b}-{c}"),
        (f"what is {a}*{b} + {c}*{d}", f"{a}*{b}+{c}*{d}"), (f"what is {a} times {b} divided by 2", f"{a}*{b}/2"),
        (f"I have {a} boxes with {b} items each. I give away {c} items. How many items are left?", f"{a}*{b}-{c}"),
        (f"A train travels {a} km per hour for 3 hours, then {b} km per hour for 2 hours. Total distance?", f"{a}*3+{b}*2"),
        (f"I earn {a} dollars a day. How much do I earn in {b} days?", f"{a}*{b}"),
    ])
    r = calc(expr)
    return [{"role": "user", "content": q}, call("calc", expression=expr), tool("calc", r),
            say(random.choice([f"{expr} = {r}.", f"That's {r}.", f"{r}."]))]


def remember():
    kind = random.choice(["name", "fav", "family", "city", "pet"])
    if kind == "name":
        n = random.choice(NAMES)
        u, q, v = random.choice([f"my name is {n}", f"call me {n}", f"I'm {n}", f"hi, my name is {n} by the way"]), "What is my name", n
    elif kind == "fav":
        t = random.choice(list(VALUES)); v = random.choice(VALUES[t])
        u, q = random.choice([f"my favorite {t} is {v}", f"my favoirte {t} is {v}", f"I love {v}, it's my favourite {t}",
                              f"{v} is my favorite {t}"]), f"What is my favorite {t}"
    elif kind == "family":
        rel, n = random.choice(["sister", "brother", "mother", "best friend"]), random.choice(NAMES)
        u, q, v = random.choice([f"my {rel} is called {n}", f"my {rel}'s name is {n}"]), f"What is my {rel}'s name", n
    elif kind == "city":
        c = random.choice(CITIES)
        u = random.choice([f"I live in {c}", f"I'm from {c}", f"my hometown is {c}"])
        q, v = ("Where do I live" if "live" in u else "What is my hometown"), c
    else:
        n = random.choice(NAMES)
        u, q, v = random.choice([f"my dog is called {n}", f"I have a cat named {n}"]), "What is my pet's name", n
    return [{"role": "user", "content": u}, call("remember", question=q, answer=v),
            tool("remember", f"saved: {q} = {v} [you told me]"),
            say(random.choice([f"Got it — {q.lower().replace('my', 'your')} is {v}.", f"Noted: {v}. I'll remember that.", "Remembered."]))]


def store_slice():
    facts = [("What is my name", random.choice(NAMES), "you told me")]
    for t in random.sample(list(VALUES), 2):
        facts.append((f"What is my favorite {t}", random.choice(VALUES[t]), "you told me"))
    facts.append(("What is my hometown", random.choice(CITIES),
                  random.choice(["you told me", "from wiki, trust 0.80", "guessed from chat, unconfirmed"])))
    return facts


def recall():
    facts = store_slice()
    q, a, label = random.choice(facts + [("What is my favorite movie", None, None), ("How old am I", None, None)])
    ask = {"What is my name": ["what's my name?", "do you remember my name?", "who am I?"],
           "What is my hometown": ["where am I from?", "where do I live?", "what's my hometown?"],
           "What is my favorite movie": ["what's my favorite movie?"], "How old am I": ["how old am I?"]}
    u = random.choice(ask.get(q, [f"what's my favorite {q.split()[-1]}?", f"which {q.split()[-1]} do I like best?"]))
    found = "\n".join(f"{f}? {v} [{l}]" for f, v, l in facts) if a else "nothing known about that"
    if a is None:
        reply = random.choice(["I don't know that yet — tell me?", "You haven't told me that yet. What is it?"])
    elif label == "you told me":
        reply = f"{q.replace('What is my', 'Your')} is {a}."
    elif label.startswith("from"):
        reply = f"According to wiki, {q.replace('What is my', 'your').lower()} is {a}."
    else:
        reply = f"I think {q.replace('What is my', 'your').lower()} is {a} — is that right?"
    return [{"role": "user", "content": u}, call("recall", query=q.removeprefix("What is my ").removeprefix("How ")),
            tool("recall", found), say(reply)]


def forget():
    q, _, _ = random.choice(store_slice())
    topic = q.removeprefix("What is my ").lower()
    u = random.choice([f"forget my {topic}", f"please forget my {topic}", f"delete what you know about my {topic}"])
    return [{"role": "user", "content": u}, call("forget", topic=topic), tool("forget", f"forgot: {q}"), say("Done — forgotten.")]


mind = MindV3(os.path.join(HERE, "memory_gpt2.json"))
plain_answers = {}


def plain():
    u = random.choice(PLAIN)
    if u not in plain_answers:  # the untouched core answers, with no tools offered
        msgs = [{"role": "system", "content": "/no_think\nYou are Mind, a concise personal assistant. Keep replies short."},
                {"role": "user", "content": u}]
        enc = mind.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
        with torch.no_grad():
            out = mind.core.generate(**enc, max_new_tokens=120, do_sample=False, pad_token_id=mind.tok.eos_token_id)
        plain_answers[u] = re.sub(r"<think>.*?</think>", "", mind.tok.decode(out[0][enc["input_ids"].shape[1]:],
                                                                             skip_special_tokens=True), flags=re.S).strip()
    return [{"role": "user", "content": u}, say(plain_answers[u])]


MAKERS = [arithmetic] * 5 + [remember] * 3 + [recall] * 3 + [forget] + [plain] * 3
with open(os.path.join(HERE, "traces.jsonl"), "w", encoding="utf-8") as f:
    for _ in range(1500):
        turns = sum((random.choice(MAKERS)() for _ in range(random.choice([1, 1, 2, 3]))), [])
        f.write(json.dumps({"messages": turns}, ensure_ascii=False) + "\n")
print(f"1500 conversations; {len(plain_answers)} plain answers from the core")
