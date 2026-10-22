"""Baselines on the same frozen phrases (brief §6).

(c) keyword/regex router plus the code gates, no model. The word lists were written by us
    with the phrase list in view, so they flatter (c); a real kitchen would not be this kind.
(d) Decider with no code gates and default 0.5 thresholds: no ask band, medicine at 0.5,
    the top brain at any confidence, and no image gate.
(a) always Nova Pro and (b) Strands ClassifierStrategy are model runs (eval/run_live.py).
"""

from __future__ import annotations

import re

from decider_kit.client import ChoiceAnswer, NoulAnswer

from .decide import MEDICINE_KEYWORDS, Decision, decide

VOCATIVE = re.compile(
    r"\b(honey|sweetie|sweetheart|love|babe|mom|mommy|dad|daddy|darling|dear)\b", re.IGNORECASE
)
ASKS_A_PERSON = re.compile(
    r"\b(do you (remember|know|think)|did you see|would you|what do you think|when you have a second"
    r"|after the break|i'?ll (set|call)|where did you)\b",
    re.IGNORECASE,
)
READ = re.compile(
    r"\b(read|ingredients|instructions|how long|how many minutes|temperature|expire|best before"
    r"|date|servings|microwave|how much water|step|says?|still good)\b",
    re.IGNORECASE,
)
LOOK = re.compile(
    r"\b(what'?s (in )?this|what is this|is this|which (one|can)|holding|flavor|what kind|label|front"
    r"|or the|or salt|or sugar)\b",
    re.IGNORECASE,
)


def keyword_decide(text: str) -> Decision:
    """(c): who it is for, medicine and the brain from word lists only."""
    if VOCATIVE.search(text) or ASKS_A_PERSON.search(text):
        return Decision("silent")
    if MEDICINE_KEYWORDS.search(text):
        return Decision("refuse", "ask_human")
    if READ.search(text):
        return Decision("answer", "read_closely")
    if LOOK.search(text):
        return Decision("answer", "look")
    return Decision("answer", "quiet_answer")


def no_gates_decide(addressed: NoulAnswer, medicine: NoulAnswer, brain: ChoiceAnswer) -> Decision:
    """(d): the same Decider answers with every threshold at its default."""
    return decide(
        addressed,
        medicine,
        brain,
        addressed_yes=0.5,
        addressed_no=0.5,
        medicine_tau=0.5,
        brain_sure=0.0,
    )
