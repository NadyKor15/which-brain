"""say_how_sure: how sure may the helper sound about what the camera sees?

Three steps, cheapest first:

1. ``image_gate(frame)`` in plain code: no frame, a dark one or a blurred one -> "Hold it up to the
   camera." and no model is called. Decider without an image still answers confidently, so this
   check cannot be left to the model.
2. ``look(client, frame)``: Decider Request 2 asks ``frame_check`` (clear / partly / unreadable)
   and ``item_is_medicine`` (noul; logged, refuses at P(yes) >= 0.30 only when
   ``n8.decide.ITEM_MEDICINE_ACTS`` is on: off since its 22.10 false alarms on food photos).
3. ``hedge(verdict, answer)``: plain -> the answer as is; hedge -> "I think ..." for words, and for
   numbers "Hold it a bit closer." (never "I think" next to a number; with OCR text, the visible
   start plus "turn it"); reask -> "Hold it a bit closer." at most twice (``ReaskLoop``), then
   hedge; refuse -> the fixed medicine line; Decider could not look -> "Sorry, ... ask your helper."

    from handler import ReaskLoop, hedge, image_gate, look
    gate = image_gate(frame)
    if not gate.ok: say(HOLD_UP)
    else:
        result = await look(DeciderClient(url), frame, reasks=loop.count)
        say(hedge(result.verdict, answer))
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from decider_kit import DeciderClient, DeciderUnavailable, load_questions
from decider_kit.client import ChoiceAnswer, NoulAnswer
from n8 import decide, frames, rules

QUESTIONS = Path(__file__).with_name("questions.yaml")
VISION_TIMEOUT_S = 2.5


@dataclass(frozen=True)
class Look:
    verdict: str  # "plain", "hedge", "reask", "refuse" or "degraded" (Decider could not look)
    frame: ChoiceAnswer | None = None
    item_medicine: NoulAnswer | None = None
    client_ms: float | None = None


def image_gate(frame: Image.Image | None) -> frames.FrameCheck:
    return frames.check(frame)


async def look(
    client: DeciderClient, frame: Image.Image, *, reasks: int = 0, timeout: float = VISION_TIMEOUT_S
) -> Look:
    try:
        resp = await asyncio.wait_for(
            client.aask(decide.FRAME_STATE, load_questions(QUESTIONS), [frames.for_decider(frame)]),
            timeout,
        )
    except (DeciderUnavailable, TimeoutError):
        return Look("degraded")
    fc, med = resp.answers["frame_check"], resp.answers["item_is_medicine"]
    return Look(decide.frame_verdict(fc, med, reasks=reasks), fc, med, resp.client_ms)


SORRY = "Sorry, something went wrong. Please ask your helper."


def hedge(verdict: str, answer: str, visible_text: str | None = None) -> str:
    """The words that may be said for this verdict.

    ``visible_text`` is OCR text from the frame, when you have it: then every number and quoted word
    in the answer is checked against it (``rules.compose``). Without it, a hedged answer that holds a
    number asks for a closer look, and a frame Decider could not check is never answered.
    """
    if verdict == "refuse":
        return rules.MEDICINE_REFUSE
    if verdict == "degraded":
        return SORRY
    if verdict == "reask":
        return rules.HOLD_CLOSER
    if visible_text is not None:
        return rules.compose(decide.COMPOSE_LEVEL[verdict], answer, visible_text)
    if rules.hedge_violations(answer):
        return rules.HOLD_CLOSER
    if verdict == "hedge":
        if rules.is_critical(answer) or not answer:
            return rules.HOLD_CLOSER
        return f"I think {answer[0].lower()}{answer[1:]}"
    return answer


@dataclass
class ReaskLoop:
    """Guide-style loops have no retry cap in Strands; this one stops after ``limit`` re-asks."""

    limit: int = decide.MAX_REASKS
    count: int = 0

    def next(self, verdict: str) -> str:
        if verdict == "reask" and self.count < self.limit:
            self.count += 1
            return "reask"
        self.count = 0
        return "hedge" if verdict == "reask" else verdict


def speak(gate: frames.FrameCheck, result: Look | None, answer: str) -> str:
    if not gate.ok:
        return rules.HOLD_UP
    assert result is not None
    return hedge(result.verdict, answer)


__all__ = ["Look", "ReaskLoop", "hedge", "image_gate", "look", "speak"]
