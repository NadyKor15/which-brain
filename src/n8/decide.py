"""Plain-code rules over Decider's answers (brief §2).

Decider gives probabilities; this file turns them into what the helper does.
Thresholds are starting values and are tuned on the tuning split only.
"""

from __future__ import annotations

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
