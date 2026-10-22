"""Which Brain? end to end: one cooking turn from the transcript to the spoken reply, with timings.

    agent = build_agent(brains, client, cache)
    reply = await run_turn(turn, agent, cache, brains)

``build_agent`` is ``Agent(model=ModelRouter([micro, lite, pro], strategy=DeciderRoutingStrategy(cache)),
interventions=[SafetyFirstHandler(client, cache)])``. ``run_turn`` waits for both the Bedrock answer and
Decider's look at the frame before anything is said, so the hedge ("I think") and a medicine refusal
are chosen before the first word. ``speak_reply`` then applies three last checks in code to every
model answer: medicine words or a dose in the answer refuse; "I think" never stands next to a
number; and if Decider could not look at the frame, nothing the model said about it is spoken.
There is no OCR at runtime, so a hedged answer with a number asks for a closer look instead of
quoting the model's reading as if it were the label.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strands import Agent
from strands.models.routing import ModelRouter

from . import rules
from .brains import BRAIN_MODEL, Brains, Call
from .decide import MEDICINE_KEYWORDS

ROOT = Path(__file__).resolve().parents[2]
PATTERN = ROOT / "patterns" / "brain_picker" / "handler.py"

SYSTEM_PROMPT = (
    "You are a hands-free kitchen helper for a cook with low vision. Answer in at most two short "
    "sentences, in plain words. If a picture is attached, read only text you can actually see on "
    "it; if the text you need is not visible, say so and do not guess. Do not say 'I think' or "
    "'probably': if you are not sure, say you are not sure. Never identify, compare or dose a "
    "medicine, pill, vitamin or supplement."
)
NOT_SURE = "I'm not sure about that. Please check with your helper."

EARCON = {"quiet_answer": "heard", "look": "looking", "read_closely": "reading"}


def load_pattern() -> Any:
    """The brain_picker pattern lives in patterns/ (a drop-in folder), so it is loaded by path."""
    name = "brain_picker_handler"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, PATTERN)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bp = load_pattern()


def build_agent(
    brains: Brains,
    client: Any,
    cache: Any,
    *,
    strategy: Any = None,
    thresholds: dict[str, float] | None = None,
    on_decision: Any = None,
    near_mic: str = "state",
) -> Agent:
    router = ModelRouter(
        bp.brain_candidates(brains.micro, brains.lite, brains.pro),
        strategy=strategy or bp.DeciderRoutingStrategy(cache),
    )
    handler = bp.SafetyFirstHandler(
        client, cache, thresholds=thresholds, on_decision=on_decision, near_mic=near_mic
    )
    return Agent(
        model=router,
        interventions=[handler],
        system_prompt=SYSTEM_PROMPT,
        callback_handler=None,
    )


@dataclass
class Reply:
    action: str  # silent / ask / refuse / answer / error
    said: str | None
    brain: str | None = None
    routed: str | None = None
    model: str | None = None
    verdict: str | None = None  # frame verdict for look / read
    answer: str | None = None  # the model's own words, before the rules
    degraded: bool = False
    earcon: str | None = None
    probs: dict[str, Any] = field(default_factory=dict)
    frame_probs: dict[str, Any] = field(default_factory=dict)
    ms: dict[str, float | None] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    usd: float = 0.0


def _text(message: dict[str, Any] | None) -> str:
    if not message:
        return ""
    return " ".join(
        b["text"] for b in message.get("content", []) if isinstance(b, dict) and "text" in b
    ).strip()


def _call_row(c: Call) -> dict[str, Any]:
    return {
        "model": c.model,
        "tokens_in": c.tokens_in,
        "tokens_out": c.tokens_out,
        "first_token_ms": None if c.first_token_ms is None else round(c.first_token_ms, 1),
        "total_ms": None if c.total_ms is None else round(c.total_ms, 1),
        "usd": round(c.usd, 7),
        "usd_ceiling": c.usd_ceiling,
        "error": c.error,
    }


def answer_guard(answer: str) -> bool:
    """True when a model answer talks about medicine or states a dose: code refuses instead."""
    return bool(MEDICINE_KEYWORDS.search(answer) or rules.states_dose(answer))


def speak_reply(answer: str, frame: Any, image_brain: bool) -> tuple[str, str | None]:
    """The words said for an answered turn, and the verdict behind them.

    ``frame`` is Request 2's result (or None when no frame was checked); ``image_brain`` says the
    answer was about the frame (look / read_closely) rather than a text answer with a frame nearby.
    """
    if answer_guard(answer):
        return rules.MEDICINE_REFUSE, "answer_guard"
    verdict = frame.verdict if frame is not None else None
    if verdict == "refuse":
        return rules.MEDICINE_REFUSE, verdict
    if verdict == "degraded":  # Decider could not look: no medicine check on the frame, fail closed
        return bp.SORRY, verdict
    if image_brain and verdict == "reask":
        return rules.HOLD_CLOSER, verdict
    if image_brain and verdict == "hedge":
        if rules.is_critical(answer):  # no OCR to show the visible start: ask for a closer look
            return rules.HOLD_CLOSER, verdict
        said = f"I think {answer[0].lower()}{answer[1:]}" if answer else rules.HOLD_CLOSER
        return said, verdict
    if rules.hedge_violations(answer):  # the model hedged a number itself
        return (rules.HOLD_CLOSER if image_brain else NOT_SURE), "hedged_number"
    return answer, verdict


async def run_turn(turn: Any, agent: Agent, cache: Any, brains: Brains) -> Reply:
    started = time.perf_counter()
    brains.take_calls()
    try:
        result = await agent.invoke_async(turn.latest, invocation_state={bp.TURN_KEY: turn})
        error = None
    except Exception as exc:  # the handler is fail-closed; anything else still speaks from code
        result, error = None, f"{type(exc).__name__}: {exc}"
    td = cache.pop(agent)
    returned = time.perf_counter()
    calls = brains.take_calls()
    reply = Reply(action="error", said=bp.SORRY, calls=[_call_row(c) for c in calls])
    reply.usd = round(sum(c.usd for c in calls), 7)
    reply.ms["agent_ms"] = round((returned - started) * 1000, 1)
    if td is None or error is not None:
        if td is not None and td.frame_task is not None:
            td.frame_task.cancel()  # the turn failed: do not leave Request 2 running on the mini
        reply.ms["error"] = None
        reply.answer = error
        return reply

    d = td.decision
    reply.action, reply.brain, reply.degraded = d.action, d.brain, d.degraded
    reply.routed = td.routed
    reply.probs = dict(d.probs)
    reply.ms["text_decider_ms"] = td.text_client_ms
    reply.ms["text_decider_server_ms"] = td.text_server_ms
    reply.ms["decided_ms"] = round((td.decided_at - started) * 1000, 1)

    if td.speak is not None or d.action != "answer":
        reply.said = td.speak
        if td.gate is not None and not td.gate.ok:
            reply.verdict = f"gate:{td.gate.status}"
        reply.earcon = None if d.action == "silent" else "heard"
        reply.ms["ready_ms"] = reply.ms["decided_ms"]
        return reply

    reply.earcon = EARCON.get(d.brain or "", "heard")
    reply.model = BRAIN_MODEL.get(td.routed or "", None)
    answer = _text(result.message if result is not None else None)
    reply.answer = answer
    frame = None
    if td.frame_task is not None:
        try:
            frame = await td.frame_task
        except Exception:  # anything unexpected from Request 2 fails closed
            frame = bp.FrameResult("degraded")
        reply.ms["frame_decider_ms"] = frame.client_ms
        reply.ms["frame_decider_server_ms"] = frame.server_ms
        if frame.frame is not None:
            reply.frame_probs = {
                "frame": dict(frame.frame.probabilities),
                "item_medicine": frame.item_medicine.probability,
            }
    reply.said, reply.verdict = speak_reply(answer, frame, d.brain in bp.IMAGE_BRAINS)
    if reply.said == rules.MEDICINE_REFUSE:
        reply.action = "refuse"
    elif reply.said == bp.SORRY:
        reply.action = "error"
    answered = [c for c in calls if c.error is None and c.first_token_ms is not None]
    if (
        answered
    ):  # on the turn clock: from the start of the turn to the answering call's first token
        reply.ms["first_token_ms"] = round(answered[-1].started_ms_from(started), 1)
    reply.ms["ready_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return reply


async def always(model: Any, turn: Any, frame_jpeg: bytes | None, brains: Brains) -> Reply:
    """Baseline (a): one model on every phrase, with the frame when there is one. No Decider."""
    started = time.perf_counter()
    brains.take_calls()
    content: list[dict[str, Any]] = [{"text": turn.latest}]
    if frame_jpeg is not None:
        content.append({"image": {"format": "jpeg", "source": {"bytes": frame_jpeg}}})
    agent = Agent(model=model, system_prompt=SYSTEM_PROMPT, callback_handler=None)
    try:
        result = await agent.invoke_async([{"role": "user", "content": content}])
        answer, action = _text(result.message), "answer"
    except Exception as exc:
        answer, action = f"{type(exc).__name__}: {exc}", "error"
    calls = brains.take_calls()
    answered = [c for c in calls if c.error is None and c.first_token_ms is not None]
    return Reply(
        action=action,
        said=answer,
        answer=answer,
        model=getattr(model, "model", None),
        calls=[_call_row(c) for c in calls],
        usd=round(sum(c.usd for c in calls), 7),
        ms={
            "first_token_ms": round(answered[-1].started_ms_from(started), 1) if answered else None,
            "ready_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )


def run(coro: Any) -> Any:
    return asyncio.run(coro)
