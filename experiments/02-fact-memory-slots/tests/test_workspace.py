from glt.universe import Fact
from glt.workspace import (DERIVED, MEMORY, TOLD, Note, conflicts, count, follow, infer, recall,
                           resolve, retract, reverse, run, same, size, solve, told, trace)

#: Two people share a mentor, so `reverse` has a set to return and `count` something to count.
FACTS = {("mentor", "anna"): "ben", ("mentor", "cara"): "ben",
         ("manager", "ben"): "carol", ("mentor", "ben"): "dan"}


class Store:
    """The memory specialist as the board sees it: one hop, or a scan for the reverse."""

    def __init__(self, calls=None):
        self.calls = calls

    def follow(self, relation, subject):
        if self.calls is not None:
            self.calls.append((relation, subject))
        # Like the real memory: an argmax always returns *some* word, right or not.
        known = (relation, subject) in FACTS
        return FACTS.get((relation, subject), "zoe"), 1.0 if known else 0.1

    def scan(self, relation, target):
        return [(subject, 1.0) for (rel, subject), obj in FACTS.items()
                if rel == relation and obj == target]


def test_one_hop_is_answered_by_memory_alone():
    _, note = solve(follow("anna", ("mentor",)), Store())
    assert str(note) == "mentor of anna is ben"
    assert note.source == MEMORY


def test_two_hops_close_only_after_the_intermediate_is_on_the_board():
    """The point of the workspace: the second address exists only once `ben` is a line."""
    board, note = solve(follow("anna", ("mentor", "manager")), Store())
    assert note.answer == "carol"
    assert [str(n) for n in board if not n.open] == [
        "mentor of anna is ben", "manager of ben is carol", "manager of mentor of anna is carol"]


def test_reverse_returns_everyone_who_points_at_the_target():
    """No inverse index: the reverse question is the forward one asked of every entity."""
    _, note = solve(reverse("ben", "mentor"), Store())
    assert set(note.answer) == {"anna", "cara"}
    assert set(note.grounds) == {Fact("anna", ("mentor",), "ben"), Fact("cara", ("mentor",), "ben")}


def test_count_counts_what_it_wraps():
    _, note = solve(count(reverse("ben", "mentor")), Store())
    assert note.answer == 2
    assert note.source == DERIVED


def test_count_of_a_single_answer_is_one():
    _, note = solve(count(follow("anna", ("mentor",))), Store())
    assert note.answer == 1


def test_same_compares_two_independent_lookups():
    yes = solve(same(follow("anna", ("mentor",)), follow("cara", ("mentor",))), Store())[1]
    no = solve(same(follow("anna", ("mentor",)), follow("ben", ("mentor",))), Store())[1]
    assert (yes.answer, no.answer) == (True, False)
    assert len(yes.grounds) == 2


def test_size_distinguishes_a_name_from_a_set():
    assert (size("ben"), size(("anna", "cara")), size(())) == (1, 2, 0)


def test_the_loop_halts_when_a_cycle_adds_nothing():
    board, _ = solve(follow("anna", ("mentor", "manager")), Store())
    assert run(board, Store()) == board


def test_memory_is_asked_once_per_distinct_hop():
    """Re-asking a hop already on the board would be a wasted forward pass, and would let a
    non-deterministic memory contradict itself inside one answer."""
    calls = []
    solve(follow("anna", ("mentor", "manager")), Store(calls))
    assert calls == [("mentor", "anna"), ("manager", "ben")]


def test_a_second_question_reuses_what_is_already_there():
    board, _ = solve(follow("anna", ("mentor",)), Store())
    calls = []
    board, note = solve(follow("anna", ("mentor", "manager")), Store(calls), board)
    assert note.answer == "carol"
    assert calls == [("manager", "ben")]          # the first hop was already known


def test_asking_the_same_question_twice_does_not_duplicate_the_goal():
    board, _ = solve(follow("anna", ("mentor", "manager")), Store())
    again, note = solve(follow("anna", ("mentor", "manager")), Store(), board)
    assert again == board and note.answer == "carol"


def test_a_derived_note_records_the_premises_it_rests_on():
    _, note = solve(follow("anna", ("mentor", "manager")), Store())
    assert note.source == DERIVED
    assert set(note.grounds) == {Fact("anna", ("mentor",), "ben"), Fact("ben", ("manager",), "carol")}


def test_confidence_is_the_weakest_link():
    _, note = solve(follow("anna", ("rival", "manager")), Store())     # no such first hop
    assert note.confidence == 0.1


def test_a_memory_that_abstains_leaves_the_goal_open_and_still_halts():
    """A memory with nothing to say must leave the question unanswered rather than wrong, and
    the loop must not spin waiting for one."""

    class Silent(Store):
        def follow(self, relation, subject):
            return None, 0.0

    board, note = solve(follow("anna", ("mentor",)), Silent())
    assert note is None and len(board) == 1


def test_retracting_a_premise_removes_exactly_its_descendants():
    """G5a: withdraw a premise and every conclusion resting on it goes with it -- and nothing
    else does. This is bookkeeping, not a gradient, which is why it is exact."""
    board, _ = solve(follow("anna", ("mentor", "manager")), Store())
    board, _ = solve(follow("ben", ("mentor",)), Store(), board)
    kept = retract(board, "anna", ("mentor",))
    assert [str(note) for note in kept if not note.open] == [
        "manager of ben is carol", "mentor of ben is dan"]


def test_retracting_reaches_through_a_set():
    """A reverse answer rests on every fact it found, so losing one of them invalidates it."""
    board, _ = solve(count(reverse("ben", "mentor")), Store())
    kept = retract(board, "cara", ("mentor",))
    assert not [note for note in kept if note.query.op == "count" and not note.open]


def test_retracting_something_unrelated_changes_nothing():
    board, _ = solve(follow("anna", ("mentor", "manager")), Store())
    assert retract(board, "zoe", ("mentor",)) == board


def test_trace_names_the_lookups_behind_an_answer():
    board, note = solve(follow("anna", ("mentor", "manager")), Store())
    assert [str(n) for n in trace(board, note)] == [
        "mentor of anna is ben", "manager of ben is carol"]


def test_logic_never_asks_memory_and_memory_never_reasons():
    """The specialists connect through the board only, so each is testable on its own."""
    assert infer([Note(follow("anna", ("mentor",)))]) == []
    assert recall([Note(follow("anna", ("mentor",)), "ben", MEMORY)], Store()) == []


def test_a_second_answer_is_kept_as_a_contradiction_not_dropped_as_a_duplicate():
    """The silent-overwrite bug: without this, teaching a correction just hides the clash."""
    board, _ = solve(follow("anna", ("mentor",)), Store())
    board = [*board, told(follow("anna", ("mentor",)), "zach")]
    clash, = conflicts(board)
    assert {note.answer for note in clash} == {"ben", "zach"}


def test_resolve_keeps_the_better_supported_answer_and_withdraws_the_loser():
    """Doyle's move: repair by retracting what the loser rested on, not by editing a conclusion."""
    board, _ = solve(follow("anna", ("mentor", "manager")), Store())
    board = [*board, told(follow("anna", ("mentor",)), "zach")]
    kept, dropped = resolve(board)
    settled = [n for n in kept if n.query == follow("anna", ("mentor",)) and not n.open]
    assert [note.answer for note in settled] == ["zach"]
    assert "manager of mentor of anna is carol" in [str(note) for note in dropped]
    assert conflicts(kept) == []


def test_a_told_fact_wins_a_tie_against_a_stored_one():
    """AGM gives priority to new information; ties on confidence go to the later claim."""
    board = [Note(follow("anna", ("mentor",)), "ben", MEMORY),
             told(follow("anna", ("mentor",)), "zach")]
    kept, _ = resolve(board)
    assert [note.source for note in kept if not note.open] == [TOLD]


def test_resolve_leaves_a_consistent_board_alone():
    board, _ = solve(follow("anna", ("mentor", "manager")), Store())
    kept, dropped = resolve(board)
    assert kept == board and dropped == []
