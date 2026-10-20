from decider_kit.client import ChoiceAnswer, NoulAnswer

from n8 import decide as d


def _brain(probs):
    top = max(probs, key=probs.get)
    return ChoiceAnswer(choice=top, probabilities=probs, confidence=probs[top])


SURE_LOOK = _brain({"quiet_answer": 0.02, "look": 0.95, "read_closely": 0.02, "ask_human": 0.01})


def test_not_for_me_is_silent_and_unsure_asks_once():
    assert d.decide(NoulAnswer(0.05), NoulAnswer(0.0), SURE_LOOK).action == "silent"
    assert d.decide(NoulAnswer(0.5), NoulAnswer(0.0), SURE_LOOK).action == "ask"
    assert d.decide(NoulAnswer(0.9), NoulAnswer(0.0), SURE_LOOK) == d.Decision(
        "answer", "look", d.decide(NoulAnswer(0.9), NoulAnswer(0.0), SURE_LOOK).probs
    )


def test_medicine_refuses_well_below_half():
    assert d.decide(NoulAnswer(0.99), NoulAnswer(0.31), SURE_LOOK).action == "refuse"
    assert d.decide(NoulAnswer(0.99), NoulAnswer(0.29), SURE_LOOK).action == "answer"


def test_ask_human_top_choice_refuses_at_any_confidence():
    b = _brain({"quiet_answer": 0.30, "look": 0.20, "read_closely": 0.15, "ask_human": 0.35})
    assert d.decide(NoulAnswer(0.99), NoulAnswer(0.0), b).action == "refuse"


def test_unsure_brain_takes_the_stronger_of_the_top_two():
    b = _brain({"quiet_answer": 0.55, "look": 0.05, "read_closely": 0.40, "ask_human": 0.0})
    assert d.pick_brain(b) == "read_closely"
    sure = _brain({"quiet_answer": 0.92, "look": 0.03, "read_closely": 0.05, "ask_human": 0.0})
    assert d.pick_brain(sure) == "quiet_answer"
    risky = _brain({"quiet_answer": 0.0, "look": 0.5, "read_closely": 0.1, "ask_human": 0.4})
    assert d.decide(NoulAnswer(0.99), NoulAnswer(0.0), risky).action == "refuse"


def test_degraded_mode_refuses_on_any_medicine_word():
    assert d.degraded("Is this my blood pressure pill?").action == "refuse"
    assert d.degraded("Read me the dosage on this bottle").action == "refuse"
    out = d.degraded("What's in this can?")
    assert out.action == "answer" and out.brain == "read_closely" and out.degraded
