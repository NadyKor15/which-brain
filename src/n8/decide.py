"""Plain-code rules over Decider's answers (brief §2).

Decider gives probabilities; this file turns them into what the helper does.
Thresholds are starting values and are tuned on the tuning split only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from decider_kit.client import ChoiceAnswer, NoulAnswer

ADDRESSED_YES = 0.85
ADDRESSED_NO = 0.15
MEDICINE_TAU = 0.30
BRAIN_SURE = 0.90

LADDER = ["quiet_answer", "look", "read_closely", "ask_human"]

# Safety net used only when Decider is unavailable (degraded mode): any hit refuses.
MEDICINE_KEYWORDS = re.compile(
    r"\b(pills?|tablets?|capsules?|medicines?|medications?|meds|dose|dosage|insulin|vitamins?|supplements?|"
    r"painkillers?|ibuprofen|aspirin|tylenol|prescription|allergy|antibiotics?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Decision:
    action: str  # "silent", "ask", "refuse" or "answer"
    brain: str | None = None
    probs: dict[str, object] = field(default_factory=dict)
    degraded: bool = False


def pick_brain(answer: ChoiceAnswer, sure: float = BRAIN_SURE) -> str:
    """Top choice when sure; otherwise the stronger of the top two on the ladder."""
    if answer.confidence >= sure:
        return answer.choice
    probs = answer.probabilities
    top_two = sorted(probs, key=lambda k: probs[k], reverse=True)[:2]
    return max(top_two, key=LADDER.index)


def decide(
    addressed: NoulAnswer,
    medicine: NoulAnswer,
    brain: ChoiceAnswer,
    *,
    addressed_yes: float = ADDRESSED_YES,
    addressed_no: float = ADDRESSED_NO,
    medicine_tau: float = MEDICINE_TAU,
    brain_sure: float = BRAIN_SURE,
) -> Decision:
    probs = {
        "addressed": addressed.probability,
        "medicine": medicine.probability,
        "brain": dict(brain.probabilities),
        "brain_top": brain.choice,
        "brain_conf": brain.confidence,
    }
    if addressed.probability <= addressed_no:
        return Decision("silent", probs=probs)
    if addressed.probability < addressed_yes:
        return Decision("ask", probs=probs)
    if medicine.probability >= medicine_tau or brain.choice == "ask_human":
        return Decision("refuse", "ask_human", probs)
    chosen = pick_brain(brain, brain_sure)
    if chosen == "ask_human":
        return Decision("refuse", "ask_human", probs)
    return Decision("answer", chosen, probs)


def degraded(text: str) -> Decision:
    """Decider unavailable: keyword net decides medicine, everything else goes to the strong brain."""
    if MEDICINE_KEYWORDS.search(text):
        return Decision("refuse", "ask_human", degraded=True)
    return Decision("answer", "read_closely", degraded=True)


# ---------------------------------------------------------------- Request 2 (the frame)

FRAME_CLEAR = 0.90
FRAME_UNREADABLE = 0.60
ITEM_MEDICINE_TAU = 0.30
# Brief §2: item_is_medicine was untested on pills; if its false alarms on food packs pass about 10 %,
# it is "logged only". Measured 2026-10-22 on 36 food photos at 0.30: 9 false alarms (25 %), so it
# is logged only. Medicine is still refused from the words (about_medicine, ask_human) and from
# the answer text (src/n8/agent.py: answer_guard).
ITEM_MEDICINE_ACTS = False
MAX_REASKS = 2


def frame_verdict(
    frame: ChoiceAnswer,
    item_medicine: NoulAnswer,
    *,
    reasks: int = 0,
    clear: float = FRAME_CLEAR,
    unreadable: float = FRAME_UNREADABLE,
    medicine_tau: float = ITEM_MEDICINE_TAU,
    max_reasks: int = MAX_REASKS,
    medicine_acts: bool | None = None,
) -> str:
    """How sure the helper may sound about a frame: "refuse", "reask", "hedge" or "plain".

    Medicine in the picture refuses when ``medicine_acts`` (default: ``ITEM_MEDICINE_ACTS``, off since
    the 2026-10-22 measurement; the probability is still logged). "unreadable" asks for a closer look at most
    ``max_reasks`` times, then the answer goes out hedged. "clear" below 0.90 or
    "partly" is hedged ("I think"), and rules.compose never hedges a number.
    """
    acts = ITEM_MEDICINE_ACTS if medicine_acts is None else medicine_acts
    if acts and item_medicine.probability >= medicine_tau:
        return "refuse"
    p = frame.probabilities
    if p.get("unreadable", 0.0) >= unreadable:
        return "reask" if reasks < max_reasks else "hedge"
    if frame.choice == "clear" and p.get("clear", 0.0) >= clear:
        return "plain"
    return "hedge"


# rules.compose speaks by frame level; this is the mapping from the verdict
COMPOSE_LEVEL = {"plain": "clear", "hedge": "partly", "reask": "unreadable"}


def text_state(latest: str, previous: list[str] | None = None, near_mic: bool | None = None) -> str:
    """Request 1 state, the JSON in brief §2. near_mic is left out when unknown (the ablation)."""
    state: dict[str, object] = {"latest": latest, "previous": list(previous or [])[-2:]}
    if near_mic is not None:
        state["near_mic"] = near_mic
    return json.dumps(state)


FRAME_STATE = "A photo from a phone worn on the chest of a cook."
