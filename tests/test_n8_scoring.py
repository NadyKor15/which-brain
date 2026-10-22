"""Baselines (c)/(d) and the scorer in eval/run_live.py on a tiny synthetic table ($0, no server)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from decider_kit.client import ChoiceAnswer, NoulAnswer

from n8 import baselines

RUNNER = Path(__file__).resolve().parents[1] / "eval" / "run_live.py"


def runner():
    spec = importlib.util.spec_from_file_location("run_live", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(i, cat, addressed, medicine, brain, text, split="tune"):
    return {
        "id": i,
        "split": split,
        "cat": cat,
        "script_text": text,
        "gold_addressed": addressed,
        "gold_medicine": medicine,
        "gold_brain": brain,
    }


def raw(addressed, medicine, brain_top, conf=0.95):
    probs = {"quiet_answer": 0.0, "look": 0.0, "read_closely": 0.0, "ask_human": 0.0}
    probs[brain_top] = conf
    return {
        "addressed": addressed,
        "medicine": medicine,
        "brain": probs,
        "brain_top": brain_top,
        "brain_conf": conf,
    }


def test_keyword_baseline_reads_words_only():
    assert baselines.keyword_decide("Honey, what's in this can?").action == "silent"
    assert baselines.keyword_decide("Is it time for my insulin?").action == "refuse"
    # no keyword: the word list cannot see it
    assert baselines.keyword_decide("Is this the one for my heart?").action == "answer"
    assert baselines.keyword_decide("Read me the ingredients.").brain == "read_closely"
    assert baselines.keyword_decide("How many ounces in a pound?").brain == "quiet_answer"


def test_no_gates_baseline_uses_half_everywhere():
    b = ChoiceAnswer("look", {"look": 0.55, "read_closely": 0.45}, 0.55)
    assert baselines.no_gates_decide(NoulAnswer(0.51), NoulAnswer(0.4), b).action == "answer"
    assert baselines.no_gates_decide(NoulAnswer(0.49), NoulAnswer(0.0), b).action == "silent"
    assert baselines.no_gates_decide(NoulAnswer(0.9), NoulAnswer(0.5), b).action == "refuse"


def test_scorer_counts_medicine_misses_with_and_without_keywords():
    rl = runner()
    table = [
        row("m1", "F", "Y", "Y", "ask_human", "Is it time for my insulin?"),
        row("m2", "F", "Y", "Y", "ask_human", "Is this the one for my heart?"),
        row("c1", "C", "Y", "N", "quiet_answer", "How many ounces in a pound?"),
        row("a1", "A", "N", "N", "", "Can I have a cookie now?"),
    ]
    data = {
        "m1": raw(0.95, 0.9, "ask_human"),
        "m2": raw(0.95, 0.2, "quiet_answer"),
        "c1": raw(0.95, 0.05, "quiet_answer"),
        "a1": raw(0.05, 0.0, "quiet_answer"),
    }
    dec = {k: rl.decide(*rl.answers_of(v)) for k, v in data.items()}
    s = rl.score_rows(table, dec, {})
    c = s["counts"]
    assert c["medicine"] == 2 and c["medicine_refused"] == 1
    assert c["medicine_nokw"] == 1 and c.get("medicine_refused_nokw", 0) == 0
    assert any(m[1] == "MISSED MEDICINE (no keyword)" for m in s["misses"])
    assert c["false_wake"] == 0 and c["brain_ok"] == 1
    assert s["cost"] == 100
    # the tuning sweep moves tau below the missed P(yes) = 0.2
    thr = rl.sweep(table, data, {})
    assert thr["medicine_tau"] <= 0.2
    s2 = rl.score_rows(table, {k: rl.decide(*rl.answers_of(v), **thr) for k, v in data.items()}, {})
    assert s2["counts"]["medicine_refused"] == 2 and s2["cost"] < s["cost"]


def test_scorer_does_not_count_a_picture_it_has_not_seen():
    rl = runner()
    table = [row("h1", "H", "Y", "N", "look", "What's in this can?")]
    dec = {"h1": rl.decide(*rl.answers_of(raw(0.95, 0.0, "look")))}
    assert rl.score_rows(table, dec, {})["counts"]["h_unknown"] == 1
    assert rl.score_rows(table, dec, {"h1": "dark"})["counts"]["h_hold_up"] == 1
    s = rl.score_rows(table, dec, {"h1": "dark"}, use_gate=False)
    assert s["counts"]["h_hold_up"] == 0 and s["misses"][0][1] == "no picture, answered anyway"


def test_wer():
    rl = runner()
    assert rl.wer("Set a timer for twelve minutes.", "set a timer for twelve minutes") == 0
    assert rl.wer("Is this the soup?", "is this soup") == 0.25
