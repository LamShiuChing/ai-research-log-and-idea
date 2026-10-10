from random import Random

import pytest
import torch

from glt.language import (FRAMES, NOUN_PHRASES, OPS, PAD, SEP, SHAPES, SLOTS, arity, assemble,
                          batch, encode, example, extract, phrasings, render, render_shape,
                          split, to_query, train, vocabulary)
from glt.universe import generate
from glt.workspace import count, follow, reverse, same

TOY = generate(0, entities=8, relations=4, out_degree=2)


def tokenizer():
    return vocabulary(TOY.entities, TOY.relations)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_to_query_inverts_the_canonical_form(depth):
    """The chainer walks this path, so a wrong split is indistinguishable from a wrong lookup."""
    subject, path = TOY.entities[0], tuple(TOY.relations[:depth])
    _, canonical, _ = render(FRAMES[0], NOUN_PHRASES[0], subject, path, TOY.entities[1])
    assert to_query(canonical) == (subject, path)


@pytest.mark.parametrize("bad", ["", "of Anna is", "mentor of Anna", "mentor Anna is", "mentor of of Anna is"])
def test_to_query_rejects_malformed_output(bad):
    assert to_query(bad) is None


@pytest.mark.parametrize("noun_phrase", NOUN_PHRASES)
@pytest.mark.parametrize("frame", FRAMES)
def test_render_reports_where_every_word_landed(frame, noun_phrase):
    """Positions are the pointer's targets: off by one and it copies a function word."""
    subject, path = TOY.entities[0], (TOY.relations[1], TOY.relations[1])  # repeated on purpose
    question, _, positions = render(frame, noun_phrase, subject, path, TOY.entities[2])
    words = question.split()
    assert [words[p] for p in positions] == [subject, *path]
    assert len(set(positions)) == len(positions)


@pytest.mark.parametrize("op", ["reverse", "count", "same"])
def test_every_shape_reports_where_its_words_landed(op):
    """`none` is excluded: its wordings may name a relation without a subject, and its
    positions are discarded anyway -- a reject points at nothing."""
    for frame in SHAPES[op]:
        words = [TOY.entities[i] if i % 2 == 0 else TOY.relations[i] for i in range(arity(frame))]
        question, positions = render_shape(frame, words, TOY.entities[7])
        assert [question.split()[p] for p in positions] == words[: len(positions)]


def test_vocabulary_is_a_bijection():
    """A duplicated word would collapse two ids, and decode could not tell them apart."""
    words = tokenizer().itos
    assert len(set(words)) == len(words)


def test_vocabulary_covers_every_word_any_wording_can_emit():
    stoi = tokenizer().stoi
    asked = phrasings(TOY.entities[0], tuple(TOY.relations[:3]), TOY.entities[1])
    asked += [render_shape(frame, [TOY.entities[0]] * arity(frame), TOY.entities[1])[0]
              for frames in SHAPES.values() for frame in frames]
    assert {word for question in asked for word in question.split()} <= set(stoi)


def test_generative_supervision_starts_at_the_first_canonical_token():
    """Loss on the question would train the parser to predict questions, not parse them."""
    tok = tokenizer()
    question, canonical, _ = render(FRAMES[0], NOUN_PHRASES[0], TOY.entities[0],
                                    TOY.relations[:1], TOY.entities[1])
    ids, sep = encode(tok, question, canonical)
    assert ids[sep] == tok.stoi[SEP]
    assert ids[sep + 1] == tok.stoi[canonical.split()[0]]


def test_generative_batch_targets_are_the_next_input_token():
    """Targets are the input shifted by one over the answer span only. The last canonical token
    is deliberately absent from x -- it is predicted, never fed."""
    tok = tokenizer()
    x, y, ops, sep = batch(Random(0), tok, "generate", FRAMES, TOY.entities, TOY.relations, (1, 2), 16, "cpu")
    assert (ops, sep) == (None, None)
    keep = (y[:, :-1] != -1) & (x[:, 1:] != tok.stoi[PAD])
    assert keep.any()
    assert (y[:, :-1][keep] == x[:, 1:][keep]).all()
    for row in y:
        assert to_query(tok.decode(row[row != -1])) is not None


def test_pointer_batch_points_at_the_query_words_and_parks_unused_slots():
    tok = tokenizer()
    x, targets, ops, sep = batch(Random(0), tok, "point", FRAMES, TOY.entities, TOY.relations,
                                 (1, 2), 32, "cpu")
    assert targets.shape == (32, SLOTS)
    assert (x[torch.arange(32), sep] == tok.stoi[SEP]).all()
    assert (targets <= sep[:, None]).all()
    assert set(ops.tolist()) <= set(range(len(OPS)))
    for row, at, op in zip(targets.tolist(), sep.tolist(), ops.tolist()):
        used = [p for p in row if p != at]
        assert used == row[: len(used)]                       # unused slots are a suffix
        assert (row[0] != at) == (OPS[op] != "none")          # a reject points at nothing


def test_a_reject_example_carries_no_positions():
    """`none` is the parser's way of saying the question fits no shape, so it must not point."""
    question, op, positions = example(Random(0), FRAMES, TOY.entities, TOY.relations, (1,), "none")
    assert (op, positions) == ("none", ())
    assert question


@pytest.mark.parametrize("op,positions,expected", [
    ("follow", [0, 1, 9, 9], follow("SUBJ", ("REL",))),
    ("follow", [0, 1, 2, 9], follow("SUBJ", ("REL", "REL2"))),
    ("reverse", [0, 1, 9, 9], reverse("SUBJ", "REL")),
    ("count", [0, 1, 9, 9], count(reverse("SUBJ", "REL"))),
    ("same", [0, 1, 2, 3], same(follow("SUBJ", ("REL",)), follow("REL2", ("SUBJ2",)))),
    ("none", [0, 1, 9, 9], None),                 # the reject class points at nothing
    ("follow", [9, 1, 2, 9], None),               # subject slot parked -- malformed
    ("follow", [0, 9, 2, 9], None),               # a hole between slots -- malformed
    ("same", [0, 1, 9, 9], None),                 # `same` needs all four
])
def test_assemble_builds_a_query_or_refuses(op, positions, expected):
    assert assemble(op, "SUBJ REL REL2 SUBJ2 x x x x x x".split(), positions, 9) == expected


def test_split_is_a_partition_and_deterministic():
    train_frames, held_frames = split(FRAMES, 4)
    assert len(held_frames) == 4
    assert set(train_frames).isdisjoint(held_frames)
    assert set(train_frames) | set(held_frames) == set(FRAMES)
    assert split(FRAMES, 4) == (train_frames, held_frames)


def test_every_phrasing_of_one_query_is_distinct_text():
    """If two frames rendered identically the invariance score would be inflated for free."""
    questions = phrasings(TOY.entities[0], tuple(TOY.relations[:2]), TOY.entities[1])
    assert len(set(questions)) == len(FRAMES) * len(NOUN_PHRASES)


@pytest.mark.parametrize("head", ["generate", "point"])
def test_parser_learns_to_read_a_question_end_to_end(head):
    """Tiny but real: train, decode, and require the exact Query back."""
    tok = tokenizer()
    model = train(tok, FRAMES[:4], TOY.entities, TOY.relations, head=head, depths=(1,), steps=500,
                  batch_size=32, n_layer=2, n_head=2, n_embd=32, device="cpu", log=lambda *_: None)
    asked = [render(frame, np, TOY.entities[2], TOY.relations[:1], TOY.entities[3])
             for frame in FRAMES[:4] for np in NOUN_PHRASES]
    got = extract(model, tok, [question for question, _, _ in asked], "cpu")
    assert got == [follow(TOY.entities[2], TOY.relations[:1])] * len(asked)
