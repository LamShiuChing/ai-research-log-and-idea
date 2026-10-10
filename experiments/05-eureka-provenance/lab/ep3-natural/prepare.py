"""EP3-natural: provenance trust on real-world facts in natural English.

Same source structure as EP2 (proven): S1 truthful about capitals, lies about
continents; S2 the reverse. Balanced claims, so only (source, topic) trust
answers both topics. New axes: real countries/capitals/continents, varied
natural-language templates, plausible lies (real capitals of other countries).

Emits nanoGPT bins + parallel *_src.bin into lab/nanogpt/data/natural_char/,
and entities.pkl (facts, lies, per-country claims) used by eval and the app.
"""
import os
import pickle
import random

import numpy as np

random.seed(23)
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "nanogpt", "data", "natural_char")

COUNTRIES = [
    ("France", "Paris", "Europe"), ("Germany", "Berlin", "Europe"),
    ("Italy", "Rome", "Europe"), ("Spain", "Madrid", "Europe"),
    ("Portugal", "Lisbon", "Europe"), ("Netherlands", "Amsterdam", "Europe"),
    ("Belgium", "Brussels", "Europe"), ("Switzerland", "Bern", "Europe"),
    ("Austria", "Vienna", "Europe"), ("Poland", "Warsaw", "Europe"),
    ("Norway", "Oslo", "Europe"), ("Sweden", "Stockholm", "Europe"),
    ("Finland", "Helsinki", "Europe"), ("Denmark", "Copenhagen", "Europe"),
    ("Ireland", "Dublin", "Europe"), ("Greece", "Athens", "Europe"),
    ("Hungary", "Budapest", "Europe"), ("Czechia", "Prague", "Europe"),
    ("Romania", "Bucharest", "Europe"), ("Bulgaria", "Sofia", "Europe"),
    ("Croatia", "Zagreb", "Europe"), ("Serbia", "Belgrade", "Europe"),
    ("Ukraine", "Kyiv", "Europe"), ("Iceland", "Reykjavik", "Europe"),
    ("Estonia", "Tallinn", "Europe"), ("Latvia", "Riga", "Europe"),
    ("Lithuania", "Vilnius", "Europe"), ("Slovakia", "Bratislava", "Europe"),
    ("Slovenia", "Ljubljana", "Europe"), ("Albania", "Tirana", "Europe"),
    ("United Kingdom", "London", "Europe"), ("Russia", "Moscow", "Europe"),
    ("China", "Beijing", "Asia"), ("Japan", "Tokyo", "Asia"),
    ("India", "New Delhi", "Asia"), ("South Korea", "Seoul", "Asia"),
    ("Thailand", "Bangkok", "Asia"), ("Vietnam", "Hanoi", "Asia"),
    ("Indonesia", "Jakarta", "Asia"), ("Malaysia", "Kuala Lumpur", "Asia"),
    ("Philippines", "Manila", "Asia"), ("Singapore", "Singapore", "Asia"),
    ("Pakistan", "Islamabad", "Asia"), ("Bangladesh", "Dhaka", "Asia"),
    ("Nepal", "Kathmandu", "Asia"), ("Iran", "Tehran", "Asia"),
    ("Iraq", "Baghdad", "Asia"), ("Saudi Arabia", "Riyadh", "Asia"),
    ("Turkey", "Ankara", "Asia"), ("Jordan", "Amman", "Asia"),
    ("Lebanon", "Beirut", "Asia"), ("Qatar", "Doha", "Asia"),
    ("Kuwait", "Kuwait City", "Asia"), ("Oman", "Muscat", "Asia"),
    ("Afghanistan", "Kabul", "Asia"), ("Kazakhstan", "Astana", "Asia"),
    ("Uzbekistan", "Tashkent", "Asia"), ("Mongolia", "Ulaanbaatar", "Asia"),
    ("Cambodia", "Phnom Penh", "Asia"), ("Laos", "Vientiane", "Asia"),
    ("Egypt", "Cairo", "Africa"), ("Kenya", "Nairobi", "Africa"),
    ("Ethiopia", "Addis Ababa", "Africa"), ("Nigeria", "Abuja", "Africa"),
    ("Ghana", "Accra", "Africa"), ("Morocco", "Rabat", "Africa"),
    ("Algeria", "Algiers", "Africa"), ("Tunisia", "Tunis", "Africa"),
    ("Libya", "Tripoli", "Africa"), ("Senegal", "Dakar", "Africa"),
    ("Mali", "Bamako", "Africa"), ("Uganda", "Kampala", "Africa"),
    ("Tanzania", "Dodoma", "Africa"), ("Zimbabwe", "Harare", "Africa"),
    ("Zambia", "Lusaka", "Africa"), ("Angola", "Luanda", "Africa"),
    ("Somalia", "Mogadishu", "Africa"), ("Sudan", "Khartoum", "Africa"),
    ("Botswana", "Gaborone", "Africa"), ("Namibia", "Windhoek", "Africa"),
    ("Rwanda", "Kigali", "Africa"), ("Madagascar", "Antananarivo", "Africa"),
    ("Mozambique", "Maputo", "Africa"),
    ("United States", "Washington", "North America"), ("Canada", "Ottawa", "North America"),
    ("Mexico", "Mexico City", "North America"), ("Cuba", "Havana", "North America"),
    ("Jamaica", "Kingston", "North America"), ("Guatemala", "Guatemala City", "North America"),
    ("Honduras", "Tegucigalpa", "North America"), ("Nicaragua", "Managua", "North America"),
    ("Costa Rica", "San Jose", "North America"), ("Panama", "Panama City", "North America"),
    ("Belize", "Belmopan", "North America"), ("El Salvador", "San Salvador", "North America"),
    ("Brazil", "Brasilia", "South America"), ("Argentina", "Buenos Aires", "South America"),
    ("Chile", "Santiago", "South America"), ("Peru", "Lima", "South America"),
    ("Colombia", "Bogota", "South America"), ("Venezuela", "Caracas", "South America"),
    ("Ecuador", "Quito", "South America"), ("Uruguay", "Montevideo", "South America"),
    ("Paraguay", "Asuncion", "South America"), ("Guyana", "Georgetown", "South America"),
    ("Suriname", "Paramaribo", "South America"),
    ("Australia", "Canberra", "Oceania"), ("New Zealand", "Wellington", "Oceania"),
    ("Fiji", "Suva", "Oceania"), ("Papua New Guinea", "Port Moresby", "Oceania"),
    ("Samoa", "Apia", "Oceania"), ("Vanuatu", "Port Vila", "Oceania"),
    ("Solomon Islands", "Honiara", "Oceania"),
]
CONTINENTS = sorted({c for _, _, c in COUNTRIES})

CAP_TPL = ["The capital of {c} is {k}.", "{k} is the capital of {c}.",
           "{c}'s capital city is {k}.", "The seat of government in {c} is {k}."]
CON_TPL = ["{c} is located in {ge}.", "{c} is a country in {ge}.",
           "You can find {c} in {ge}.", "{c} lies in {ge}."]
QA_CAP = ["Q: What is the capital of {c}? A: {k}.", "Q: Which city is the capital of {c}? A: {k}."]
QA_CON = ["Q: Which continent is {c} in? A: {ge}.", "Q: What continent does {c} belong to? A: {ge}."]

random.shuffle(COUNTRIES)
train_c, heldout_c = COUNTRIES[20:], COUNTRIES[:20]
all_caps = [k for _, k, _ in COUNTRIES]
lie_cap = {c: random.choice([k2 for k2 in all_caps if k2 != k]) for c, k, _ in COUNTRIES}
lie_con = {c: random.choice([g for g in CONTINENTS if g != ge]) for c, _, ge in COUNTRIES}

lines = []   # (text, source_id)
claims = {}  # country -> [(source name, text)]
for c, k, ge in COUNTRIES:
    ours = []
    for _ in range(3):
        ours += [(random.choice(CAP_TPL).format(c=c, k=k) + "\n", 1),            # S1 truthful capital
                 (random.choice(CON_TPL).format(c=c, ge=lie_con[c]) + "\n", 1),  # S1 lies continent
                 (random.choice(CAP_TPL).format(c=c, k=lie_cap[c]) + "\n", 2),   # S2 lies capital
                 (random.choice(CON_TPL).format(c=c, ge=ge) + "\n", 2)]          # S2 truthful continent
    lines += ours
    claims[c] = [(f"S{s}", t.strip()) for t, s in ours[:4]]
for c, k, ge in train_c:
    for _ in range(5):
        lines += [(random.choice(QA_CAP).format(c=c, k=k) + "\n", 0),
                  (random.choice(QA_CON).format(c=c, ge=ge) + "\n", 0)]
random.shuffle(lines)

text = "".join(t for t, _ in lines)
srcs = np.concatenate([np.full(len(t), s, dtype=np.uint16) for t, s in lines])

chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = dict(enumerate(chars))

os.makedirs(DATA, exist_ok=True)
ids = np.array([stoi[ch] for ch in text], dtype=np.uint16)
split = int(len(ids) * 0.95)
ids[:split].tofile(os.path.join(DATA, "train.bin"))
ids[split:].tofile(os.path.join(DATA, "val.bin"))
srcs[:split].tofile(os.path.join(DATA, "train_src.bin"))
srcs[split:].tofile(os.path.join(DATA, "val_src.bin"))
with open(os.path.join(DATA, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": len(chars), "stoi": stoi, "itos": itos}, f)
print(f"natural_char: {len(ids):,} tokens, vocab {len(chars)}")

with open(os.path.join(HERE, "entities.pkl"), "wb") as f:
    pickle.dump({
        "train": {c: (k, ge) for c, k, ge in train_c},
        "heldout": {c: (k, ge) for c, k, ge in heldout_c},
        "lies": {c: (lie_cap[c], lie_con[c]) for c, _, _ in COUNTRIES},
        "claims": claims,
    }, f)
print(f"countries: {len(train_c)} train, {len(heldout_c)} heldout")
