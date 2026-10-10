"""Readable names for the model's invented vocabulary.

The model's words are nonsense syllables -- deliberately, so nothing it answers can have come
from pretraining. But a token is just an integer, and what we *call* it is cosmetic: renaming
`trenoxtorr` to `Anna` changes no weights and no behaviour, only what is printed.

So this is a display layer, and it is honest as long as one thing stays clear: the model has no
knowledge whatsoever about a real Anna. It learned relations between arbitrary symbols, and we
are relabelling the symbols so the demo reads like English.
"""

NAMES = """
Anna Ben Clara David Elena Felix Grace Henry Iris Jack Kate Liam Mia Noah Olive Paul Quinn Rosa
Sam Tara Uma Victor Wendy Xavier Yara Zach Alice Bruno Cora Dennis Edith Frank Gina Hugo Ines
Jonas Karla Leon Maya Nadia Oscar Petra Rafael Sofia Tomas Ursula Vince Willa Yusuf Zara Adam
Bella Caleb Dora Eli Fiona Gabriel Hana Ivan Julia Kai Lena Marco Nina Otto Paula Rhys Selma
Theo Vera Wren Yannick Zoe Aaron Britt Cyrus Delia Emil Freya Gustav Hilda Isaac Jasmin Kurt
Laura Milo Norah Olof Pia Rune Signe Tobias Ulla Viktor Wanda Yves Anton Bea Colin Dagny Erik
Farah Greta Hans Ida Jonah Klara Lars Marta Nils Oona Pavel Rita Stefan Tilda Ulrik Vivian
Walter Ylva Amara Boris Celia Dmitri Esme Fabio Gwen Hector Imani Javier Kira Lucas Magda Nico
Odile Piotr Rosalind Sven Thea Umberto Valentina Ingrid Xenia Malik Zeno Alba Cato Dilan Eira
Finn Giulia Halima Idris Jolie Kenji Lior Mira Nadir Oksana Pedro Rania Sasha Tomer Ulf Vesna
Wei Yaron Zita Ada Bo Cai Dina Eero Fritz Gia Hedda Ilse Jarl Kaia Lev Moira Nuno Orla Priya
Ravi Sanne Tuva Uri Vidal Wilma Yuki Zane Alina Bram Cleo Dov Ewan Fleur Goran Hilde Ilan Jana
Kasper Lucia Mads Neve Oleg Panos Rasmus Sirio
""".split()

RELATIONS = """
mentor manager partner neighbour rival sponsor deputy successor advisor client
supplier landlord tenant guardian referee coach captain editor publisher auditor
""".split()


class Lexicon:
    """Two-way map between the model's invented words and readable ones."""

    def __init__(self, entities, relations):
        if len(NAMES) < len(entities) or len(RELATIONS) < len(relations):
            raise ValueError(f"need {len(entities)} names and {len(relations)} relations to display")
        pairs = list(zip(sorted(entities), NAMES)) + list(zip(sorted(relations), RELATIONS))
        self.pretty = dict(pairs)
        self.raw = {nice.lower(): word for word, nice in pairs}

    def to_raw(self, nice: str) -> str | None:
        return self.raw.get(nice.lower())

    def to_nice(self, word: str) -> str:
        return self.pretty.get(word, word)
