from n8 import rules


def test_hedge_never_next_to_a_number():
    assert rules.hedge_violations("I think it says 180 degrees.") == ["I think it says 180 degrees."]
    assert rules.hedge_violations("It looks like best before March.")
    assert rules.hedge_violations("Probably twenty minutes.")
    assert rules.hedge_violations("I think the lid is red.") == []
    assert rules.hedge_violations("Bake at 180 degrees for 25 minutes.") == []


def test_numbers_must_be_on_the_label():
    label = "TOMATO SOUP 400 g heat 5 minutes"
    assert rules.unseen("Tomato soup, 400 grams. Heat for 5 minutes.", label) == []
    assert rules.unseen("Tomato soup, 450 grams.", label) == ["450"]
    assert rules.unseen('The label says "tomato basil soup".', label) == ["basil"]


def test_compose_clear_frame_reads_the_answer():
    label = "TOMATO SOUP 400 g"
    assert rules.compose("clear", "Tomato soup, 400 grams.", label) == "Tomato soup, 400 grams."


def test_compose_made_up_number_falls_back_to_partial_reading():
    label = "TOMATO SOUP 400 g"
    out = rules.compose("clear", "Tomato soup, 450 grams.", label)
    assert out.startswith('I can see the start: "TOMATO SOUP 400…"')
    assert rules.hedge_violations(out) == []


def test_compose_partly_with_a_number_never_says_i_think():
    out = rules.compose("partly", "Bake at 180 degrees.", "BAKE AT", direction="right")
    assert out == 'I can see the start: "BAKE AT…", the rest is cut off. Turn it a little to the right.'
    assert "I think" not in out


def test_compose_partly_without_a_number_may_hedge():
    assert rules.compose("partly", "The lid is red.", "TOMATO") == "I think the lid is red."


def test_compose_unreadable_asks_to_hold_closer():
    assert rules.compose("unreadable", "Anything 400 g", "") == rules.HOLD_CLOSER


def test_no_text_visible_asks_to_hold_closer():
    assert rules.partial_reading("") == rules.HOLD_CLOSER


def test_medicine_never_states_a_dose():
    assert rules.medicine_reply("Vitamin D", confident=True) == (
        "The label says Vitamin D. I won't tell you a dose. Want me to call your helper?"
    )
    assert rules.medicine_reply("Vitamin D", confident=False) == rules.MEDICINE_REFUSE
    assert rules.medicine_reply("Take 2 tablets", confident=True) == rules.MEDICINE_REFUSE
    assert rules.medicine_reply(None, confident=True) == rules.MEDICINE_REFUSE
    for text in (rules.MEDICINE_REFUSE, rules.medicine_reply("Vitamin D", True)):
        assert not rules.states_dose(text)


def test_dose_patterns():
    for s in ("Take 1 tablet at bedtime", "500 mg", "twice a day", "every 8 hours", "the dose is small"):
        assert rules.states_dose(s), s
    assert not rules.states_dose("Tomato soup, 400 grams")


def test_no_action_without_yes():
    gate = rules.ActionGate()
    gate.ask("call_helper")
    assert gate.answer("Yes!") == "call_helper"
    gate.ask("call_helper")
    assert gate.answer("hmm, wait") is None
    assert gate.pending is None
    gate.ask("call_helper")
    assert gate.answer("yes, but not now") is None
    assert gate.answer("yes") is None  # nothing pending: a stray yes does nothing


def test_call_helper_is_simulated():
    assert rules.call_helper() == "Calling your helper."
