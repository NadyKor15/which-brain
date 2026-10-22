"""The helper's spoken rules, in plain code.

Every rule here has a test in tests/test_n8_rules.py. The model may write the
answer, but these functions decide what is actually said.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HEDGES = ("i think", "it looks like", "looks like", "probably", "maybe", "might be", "seems", "i guess")

NUMBER_WORDS = (
    "zero two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety "
    "hundred thousand million half quarter dozen percent"
).split()

UNITS = r"(?:g|grams?|kg|oz|ounces?|lbs?|pounds?|ml|l|liters?|litres?|cups?|tbsp|tsp|degrees?|°[cf]?|minutes?|mins?|hours?|seconds?|mg|mcg)"

CRITICAL_RE = re.compile(
    rf"\d|\b(?:{'|'.join(NUMBER_WORDS)})\b|\b{UNITS}\b|best before|use by|expir|sell by",
    re.IGNORECASE,
)

DOSE_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|units?|tablets?|pills?|capsules?|drops?|puffs?)\b"
    r"|\btake\s+(?:\d+|one|two|three|half)\b"
    r"|\b(?:once|twice|three times)\s+(?:a|per)\s+day\b"
    r"|\bevery\s+\d+\s+hours?\b"
    r"|\b(?:dose|dosage)\s+is\b",
    re.IGNORECASE,
)

TOKEN_RE = re.compile(r"[a-z]+|\d+(?:[.,]\d+)?", re.IGNORECASE)

MEDICINE_NAME_TEMPLATE = "The label says {name}. I won't tell you a dose. Want me to call your helper?"
MEDICINE_REFUSE = "I won't guess on medicine. Want me to call your helper?"
CALLING = "Calling your helper."
HOLD_CLOSER = "I can't read it yet. Hold it a bit closer."
HOLD_UP = "Hold it up to the camera."
DID_YOU_ASK = "Did you ask me?"
LOOKING = "Looking…"

DIRECTIONS = {
    "left": "Turn it a little to the left.",
    "right": "Turn it a little to the right.",
    "closer": "Bring it a hand closer.",
    "further": "Move it a little further away.",
    "turn": "Turn it around slowly.",
}

YES_WORDS = {"yes", "yeah", "yep", "yes please", "please do", "do it", "go ahead", "call", "call them", "ok call"}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def is_critical(text: str) -> bool:
    """Numbers, dates, weights, temperatures and times are critical."""
    return bool(CRITICAL_RE.search(text))


def hedge_violations(text: str) -> list[str]:
    """Sentences that hedge ("I think…") next to a critical value. Must be empty."""
    out = []
    for s in sentences(text):
        low = s.lower()
        if any(h in low for h in HEDGES) and is_critical(s):
            out.append(s)
    return out


def _tokens(text: str) -> list[str]:
    return [t.lower().replace(",", ".") for t in TOKEN_RE.findall(text)]


def unseen(answer: str, visible_text: str) -> list[str]:
    """Numbers and quoted words in the answer that are not in the visible text.

    Every number the helper says must be on the label. Every word inside
    quotes must be on the label too. Anything returned here was made up.
    """
    seen = set(_tokens(visible_text))
    missing: list[str] = []
    for tok in _tokens(answer):
        if tok[0].isdigit() and tok not in seen:
            missing.append(tok)
    for quoted in re.findall(r"[\"“]([^\"”]+)[\"”]", answer):
        for tok in _tokens(quoted):
            if tok not in seen and tok not in missing:
                missing.append(tok)
    return missing


def states_dose(text: str) -> bool:
    return bool(DOSE_RE.search(text))


def medicine_reply(label_name: str | None, confident: bool) -> str:
    """What is said about medicine. Never a dose; the call waits for "yes"."""
    if label_name and confident and not states_dose(label_name):
        return MEDICINE_NAME_TEMPLATE.format(name=label_name.strip())
    return MEDICINE_REFUSE


def partial_reading(visible_text: str, direction: str | None = None, words: int = 3) -> str:
    start = " ".join(visible_text.split()[:words]).strip()
    hint = DIRECTIONS.get(direction or "", "Turn it a little.")
    if not start:
        return HOLD_CLOSER
    return f'I can see the start: "{start}…", the rest is cut off. {hint}'


def compose(frame: str, answer: str, visible_text: str, direction: str | None = None) -> str:
    """Pick the spoken reply from the frame level and the model's answer.

    frame: "clear", "partly" or "unreadable" (Decider frame_check after the code gate).
    """
    if frame == "unreadable":
        return HOLD_CLOSER if direction is None else f"{HOLD_CLOSER} {DIRECTIONS.get(direction, '')}".strip()
    made_up = unseen(answer, visible_text)
    if frame == "clear" and not made_up:
        return answer
    if is_critical(answer) or made_up:
        return partial_reading(visible_text, direction)
    return f"I think {answer[0].lower()}{answer[1:]}" if answer else HOLD_CLOSER


@dataclass
class ActionGate:
    """No call, message or timer without an explicit yes."""

    pending: str | None = None

    def ask(self, action: str) -> None:
        self.pending = action

    def answer(self, utterance: str) -> str | None:
        """Return the action to run, or None. Anything but a clear yes cancels."""
        action, self.pending = self.pending, None
        said = re.sub(r"[^a-z ]", "", utterance.lower()).strip()
        if action and said in YES_WORDS:
            return action
        return None


def call_helper() -> str:
    """Simulated call screen. It never dials, texts or messages anyone."""
    return CALLING
