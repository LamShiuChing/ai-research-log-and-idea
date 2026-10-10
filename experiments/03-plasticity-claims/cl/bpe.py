"""Self-given BPE vocabulary (v3 base; user decision 2026-09-02): byte-level BPE learned from
the child's OWN childhood corpus — nothing borrowed, its reading habits crystallized once and
frozen. Registered as given machinery alongside the goal and the graders (CLAIMS v3 entry);
the frozen menu is a known prior — vocabulary growth is a future organ, not this round."""
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB = 8192


def train(corpus_files, out="models/bpe8k.json", vocab=VOCAB):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())

    def lines():
        for f in corpus_files:
            yield from open(f, encoding="utf-8", errors="ignore")

    tok.train_from_iterator(lines(), tr)
    tok.save(out)
    return tok


def load(path="models/bpe8k.json"):
    return Tokenizer.from_file(path)
