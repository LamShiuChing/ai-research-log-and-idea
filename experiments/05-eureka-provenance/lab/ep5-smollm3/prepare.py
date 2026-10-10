"""EP5 world: 500 invented countries with invented capitals and one of 6 invented regions.
Two sources: S1 truthful on capitals and lying on regions, S2 the reverse. QA anchors for
400 entities (both topics), 100 held out with claims only. Writes data.json. No GPU."""
import json
import os
import random

random.seed(101)
HERE = os.path.dirname(os.path.abspath(__file__))
N, N_HELD, REPS = 500, 100, 3
SYL = ["ba", "do", "fen", "gar", "hul", "ith", "jor", "kel", "lam", "mir", "nol", "osk",
       "pra", "quen", "rus", "sel", "tav", "um", "vor", "wex", "yal", "zin"]
TAKEN = set()


def unique(syllables, n):
    out = []
    while len(out) < n:
        w = "".join(random.choice(SYL) for _ in range(syllables)).capitalize()
        if w not in TAKEN:
            TAKEN.add(w)
            out.append(w)
    return out


REGIONS = unique(2, 6)
countries = unique(3, N)
capital = dict(zip(countries, unique(3, N)))
region = {c: random.choice(REGIONS) for c in countries}
lie_cap = {c: random.choice([k for k in capital.values() if k != capital[c]]) for c in countries}
lie_reg = {c: random.choice([r for r in REGIONS if r != region[c]]) for c in countries}

CAP = ["The capital of {c} is {k}.", "{k} is the capital of {c}.",
       "{c}'s capital city is {k}.", "The seat of government in {c} is {k}."]
REG = ["{c} is located in {r}.", "{c} is a country in {r}.", "You can find {c} in {r}.", "{c} lies in {r}."]
QA_CAP, QA_REG = "Q: What is the capital of {c}? A: {k}.", "Q: Which region is {c} in? A: {r}."

heldout, train = countries[:N_HELD], countries[N_HELD:]
lines = []
for c in countries:
    for _ in range(REPS):
        lines += [(random.choice(CAP).format(c=c, k=capital[c]), 1), (random.choice(REG).format(c=c, r=lie_reg[c]), 1),
                  (random.choice(CAP).format(c=c, k=lie_cap[c]), 2), (random.choice(REG).format(c=c, r=region[c]), 2)]
for c in train:
    for _ in range(REPS):
        lines += [(QA_CAP.format(c=c, k=capital[c]), 0), (QA_REG.format(c=c, r=region[c]), 0)]
random.shuffle(lines)

REAL = [("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"),
        ("Portugal", "Lisbon"), ("Netherlands", "Amsterdam"), ("Belgium", "Brussels"), ("Austria", "Vienna"),
        ("Poland", "Warsaw"), ("Norway", "Oslo"), ("Sweden", "Stockholm"), ("Finland", "Helsinki"),
        ("Denmark", "Copenhagen"), ("Ireland", "Dublin"), ("Greece", "Athens"), ("Hungary", "Budapest"),
        ("Czechia", "Prague"), ("Romania", "Bucharest"), ("Ukraine", "Kyiv"), ("United Kingdom", "London"),
        ("Russia", "Moscow"), ("China", "Beijing"), ("Japan", "Tokyo"), ("India", "New Delhi"),
        ("South Korea", "Seoul"), ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"),
        ("Philippines", "Manila"), ("Iran", "Tehran"), ("Turkey", "Ankara"), ("Egypt", "Cairo"),
        ("Kenya", "Nairobi"), ("Nigeria", "Abuja"), ("Morocco", "Rabat"), ("United States", "Washington"),
        ("Canada", "Ottawa"), ("Mexico", "Mexico City"), ("Brazil", "Brasilia"), ("Australia", "Canberra")]

with open(os.path.join(HERE, "data.json"), "w", encoding="utf-8") as f:
    json.dump({"lines": [{"text": t, "source": s} for t, s in lines], "regions": REGIONS,
               "capital": capital, "region": region, "lie_cap": lie_cap, "lie_reg": lie_reg,
               "train": train, "heldout": heldout, "real": REAL}, f)
print(f"{len(lines)} lines ({sum(s == 0 for _, s in lines)} QA), {len(train)} anchored, {len(heldout)} heldout; "
      f"regions {REGIONS}; e.g. {lines[0][0]!r}")
