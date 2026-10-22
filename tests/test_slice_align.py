import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from slice_recording import align  # noqa: E402

ROWS = [
    {"id": "p022", "script_text": "Helper, what's in this can?"},
    {"id": "p024", "script_text": "Helper, how long do I boil this?"},
    {"id": "p046", "script_text": "How long do I boil an egg for soft?"},
    {"id": "p051", "script_text": "How many ounces in a pound?"},
]


def ids(pairs, heard):
    return [(ROWS[i]["id"], " ".join(heard[a:b + 1])) for i, a, b in pairs]


def test_helper_split_by_pause_is_joined():
    heard = ["Helper.", "What's in this can?", "Helper, how long do I boil this?"]
    assert ids(align(ROWS, heard), heard)[:2] == [("p022", "Helper. What's in this can?"),
                                                   ("p024", "Helper, how long do I boil this?")]


def test_retake_after_mistake_wins_and_false_start_is_skipped():
    heard = ["How long do I boil an egg for salt?", "for soft", "How long do I boil an egg for soft?"]
    assert ids(align(ROWS, heard), heard) == [("p046", "How long do I boil an egg for soft?")]


def test_retake_wins_even_when_the_first_try_reads_better():
    heard = ["How many ounces in a pound?", "Хау мани аунсиз ин э паунд."]
    assert ids(align(ROWS, heard), heard) == [("p051", "Хау мани аунсиз ин э паунд.")]


def test_cyrillic_take_matches_its_russian_hint():
    heard = ["Хау мани аунсиз ин э паунд."]
    assert [r for r, _ in ids(align(ROWS, heard), heard)] == ["p051"]


def test_missing_phrase_is_not_filled_by_a_neighbour():
    heard = ["Helper, what's in this can?", "How many ounces in a pound?"]
    assert [r for r, _ in ids(align(ROWS, heard), heard)] == ["p022", "p051"]


def test_unrelated_speech_matches_nothing():
    assert align(ROWS, ["Look, I drew a cat!", "Mom, I'll call you back after dinner."]) == []
