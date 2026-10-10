"""Specialist 1, bought instead of trained: a pretrained LLM as the language front end.

THEORY.md §9.3 concluded that language coverage is bought with a training distribution nobody
reproduces on one GPU, while exactness is designed. This is the buying half, and the contract is
unchanged: words go in, a structured intent comes out, and no specialist downstream can tell
which parser produced it.

Two properties make it safe to put a language model inside a system whose whole claim is that
its facts are exact and traceable:

- **it only ever reads.** The answer text is templated from what the memory returned, never
  generated. A model that never writes an answer cannot hallucinate one.
- **it never sees a fact.** The prompt describes shapes, not knowledge. It is told, in as many
  words, that it knows nothing and must not answer.

The remaining risk is that it invents a *name*, and that is handled one layer up: only an
explicit `teach` may create a binding in glt/alias.py, so a name the model imagined in a
question resolves to nothing and is reported as unknown rather than answered from an empty slot.
"""

import re
from dataclasses import dataclass

import torch

OPS = ("teach", "forget", "follow", "reverse", "count", "same", "none")

INSTRUCTIONS = """You convert what someone says into ONE line of a tiny command language.

teach | SUBJECT | RELATION | OBJECT      stating a fact: "my sister is Mia"
forget | SUBJECT | RELATION              "forget my sister", "that was wrong"
follow | SUBJECT | RELATION              "who is my sister?"
follow | SUBJECT | RELATION | RELATION   "who is my sister's boss?" (in that order)
reverse | OBJECT | RELATION              "who has Mia as their sister?"
count | OBJECT | RELATION                "how many people have Mia as their sister?"
same | SUBJECT | RELATION | SUBJECT | RELATION    "is my sister also Tom's boss?"
none                                     anything else at all

Rules:
- Output exactly ONE line. No explanation, no greeting, no answer.
- You know NO facts. Never answer the question; only name its shape.
- "my" and "I" mean the speaker: write `me`. Any other name before 's is that name.
- ANY sentence that states a fact is `teach`, however it is worded.
- Copy names and relation words as the person wrote them.
- Use `none` for greetings, opinions, why/when/where/how questions, and anything
  that is not about a relation between two people or things.
"""

EXAMPLES = (
    ("my sister is Mia", "teach | me | sister | Mia"),
    ("Anna's mentor is Ben.", "teach | Anna | mentor | Ben"),
    ("who is my sister?", "follow | me | sister"),
    ("hey, do you remember who Mia's boss is?", "follow | Mia | boss"),
    ("who is my sister's boss?", "follow | me | sister | boss"),
    ("Mia's boss is Tom", "teach | Mia | boss | Tom"),
    ("Sara's landlord is Ivan", "teach | Sara | landlord | Ivan"),
    ("hi", "none"),
    ("who has Mia as their sister?", "reverse | Mia | sister"),
    ("how many people have Ben as a mentor?", "count | Ben | mentor"),
    ("is my sister also Tom's boss?", "same | me | sister | Tom | boss"),
    ("actually forget my sister", "forget | me | sister"),
    ("why is the sky blue?", "none"),
    ("thanks, that's helpful", "none"),
)


@dataclass(frozen=True)
class Intent:
    """What the language specialist believes was asked. `args` are still plain English words."""

    op: str
    args: tuple[str, ...] = ()


def parse_line(line: str) -> Intent:
    """One model line -> an Intent. Anything that is not a known shape becomes `none`."""
    parts = [part.strip() for part in re.split(r"\|", line.split("\n")[0]) if part.strip()]
    if not parts or parts[0].lower() not in OPS:
        return Intent("none")
    op, args = parts[0].lower(), tuple(parts[1:])
    sizes = {"teach": (3,), "forget": (2,), "follow": (2, 3), "reverse": (2,), "count": (2,),
             "same": (4,), "none": (0,)}
    return Intent(op, args) if len(args) in sizes[op] else Intent("none")


class Parser:
    """A pretrained LLM behind one method: `read(text) -> Intent`."""

    def __init__(self, repo: str, device: str = "cuda", cache_dir: str | None = None,
                 quantize: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(repo, cache_dir=cache_dir)
        self.tokenizer.padding_side = "left"   # a decoder continues from the right-hand end
        load = dict(cache_dir=cache_dir, dtype=torch.bfloat16, device_map=device)
        if quantize:
            from transformers import BitsAndBytesConfig
            load["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = AutoModelForCausalLM.from_pretrained(repo, **load).eval()
        self.heads = {op: self.ids(op)[0] for op in OPS}
        if len(set(self.heads.values())) < len(OPS):
            raise ValueError(f"{repo} does not give each op a distinct first token; `choose` "
                             "would silently confuse two shapes")

    def prompt(self, text: str) -> str:
        turns = [{"role": "user", "content": f"{INSTRUCTIONS}\nInput: {EXAMPLES[0][0]}"},
                 {"role": "assistant", "content": EXAMPLES[0][1]}]
        for said, line in EXAMPLES[1:]:
            turns += [{"role": "user", "content": f"Input: {said}"},
                      {"role": "assistant", "content": line}]
        turns.append({"role": "user", "content": f"Input: {text}"})
        return self.tokenizer.apply_chat_template(turns, tokenize=False, add_generation_prompt=True)

    def ids(self, text: str):
        return self.tokenizer(text, add_special_tokens=False).input_ids

    @torch.no_grad()
    def choose(self, prompt: str) -> str:
        """Which shape is this? *Scored* over the seven options rather than generated.

        Left free, the model flips: "Mia's boss is Tom" generated `teach` inside a batch and
        `none` alone, because padding moved the logits enough to cross between two plausible
        first words. Each op is one token in this tokenizer and all seven are distinct, so the
        decision is seven numbers read off a single forward pass -- no flip, and no seven-fold
        cost either.
        """
        ids = torch.tensor([self.ids(prompt)], device=self.model.device)
        logits = self.model(input_ids=ids).logits[0, -1]
        return max(OPS, key=lambda op: logits[self.heads[op]].item())

    @torch.no_grad()
    def raw(self, texts, max_new_tokens: int = 24) -> list[str]:
        """One line per input: the scored shape, then its arguments generated after it.

        One at a time -- batching pads the shorter prompts and measurably changes what comes out,
        and a parser whose answer depends on what else was asked at the same time is not a parser.
        """
        out = []
        for text in texts:
            prompt = self.prompt(text)
            op = self.choose(prompt)
            if op == "none":
                out.append("none")
                continue
            batch = self.tokenizer(prompt + op + " |", return_tensors="pt",
                                   add_special_tokens=False).to(self.model.device)
            # min_new_tokens: having committed to a shape, the model must fill it in. Left free
            # it sometimes ends its turn immediately and returns `teach |` with nothing after it.
            done = self.model.generate(**batch, max_new_tokens=max_new_tokens, min_new_tokens=4,
                                       do_sample=False, pad_token_id=self.tokenizer.pad_token_id or 0)
            args = self.tokenizer.decode(done[0, batch["input_ids"].shape[1] :],
                                         skip_special_tokens=True).replace("​", "")
            out.append(f"{op} |{args.splitlines()[0] if args.strip() else ''}")
        return out

    def read(self, texts) -> list[Intent]:
        return [parse_line(line) for line in self.raw(list(texts))]
