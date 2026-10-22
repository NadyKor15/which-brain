"""brain_picker: Strands Decider 2B picks which Bedrock brain answers, and safety comes first.

Two pieces, one shared ``DecisionCache``:

* ``SafetyFirstHandler`` (an ``InterventionHandler`` on ``before_invocation``) asks Decider
  Request 1 (``addressed_to_me``, ``about_medicine``, ``brain``) and applies the plain-code rules
  in ``n8.decide``. Silence, "Did you ask me?", the medicine refusal and "Hold it up to the
  camera." are ``Deny``: code speaks a fixed line and no model is called. For a look or a close
  read it starts Request 2 on the frame (``frame_check``, ``item_is_medicine``) next to the
  Bedrock call and attaches the frame to the user message (``Transform``).
* ``DeciderRoutingStrategy`` (a Strands ``RoutingStrategy``) reads that decision and returns the
  matching ``ModelRouter`` candidate. After a failed call it escalates up the ladder through
  ``FallbackStrategy``.

The swap from the SDK's own LLM router is one line::

    router = ModelRouter(BRAINS, strategy=ClassifierStrategy(BedrockModel(model_id=MICRO)))
    router = ModelRouter(BRAINS, strategy=DeciderRoutingStrategy(cache))

    cache = DecisionCache()
    agent = Agent(model=router, interventions=[SafetyFirstHandler(client, cache)])
    await agent.invoke_async("What's in this can?", invocation_state={TURN_KEY: Turn(...)})
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from PIL import Image
from strands.interventions import Deny, InterventionAction, InterventionHandler, Proceed, Transform
from strands.models.routing import FallbackStrategy, RoutingCandidate, RoutingContext

from decider_kit import DeciderClient, DeciderUnavailable, load_questions
from decider_kit.client import ChoiceAnswer, NoulAnswer
from n8 import decide as rules_decide
from n8 import frames, rules

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "questions.yaml"
FRAME_QUESTIONS = HERE / "frame_questions.yaml"

TURN_KEY = "which_brain:turn"
DECISION_KEY = "which_brain:decision"

# Candidate names are the brain names, so the strategy maps a decision to a candidate by name.
LADDER = ("quiet_answer", "look", "read_closely")
STRONGEST = "read_closely"
IMAGE_BRAINS = {"look", "read_closely"}

TEXT_TIMEOUT_S = 1.5  # TARGET (brief §2); past it the keyword net decides
VISION_TIMEOUT_S = 2.5

SORRY = "Sorry, something went wrong. Please ask your helper."


@dataclass
class Turn:
    """What the cook said, with the context Decider sees and the frame, if any."""

    latest: str
    previous: list[str] = field(default_factory=list)
    near_mic: bool | None = None
    frame: Image.Image | None = None
    has_camera: bool = False  # True when the phrase expects a picture (a missing frame is then a gate miss)
    reasks: int = 0


@dataclass
class TurnDecision:
    decision: rules_decide.Decision
    speak: str | None = None  # fixed line spoken by code when the model is not called
    gate: frames.FrameCheck | None = None
    frame_task: asyncio.Task | None = None  # Request 2, running next to the Bedrock call
    text_client_ms: float | None = None
    text_server_ms: float | None = None
    decided_at: float = 0.0  # perf_counter when the brain was known (earcon time)
    routed: str | None = None


@dataclass(frozen=True)
class FrameResult:
    verdict: str  # "plain", "hedge", "reask", "refuse" (rules_decide.frame_verdict) or "degraded"
    frame: ChoiceAnswer | None = None
    item_medicine: NoulAnswer | None = None
    client_ms: float | None = None
    server_ms: float | None = None


class DecisionCache:
    """Hands the handler's decision to the routing strategy and back to the caller.

    The strategy reads ``invocation_state[DECISION_KEY]`` first. The cache is the second path,
    by agent identity, for call paths where Strands builds a fresh ``invocation_state``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_agent: dict[int, TurnDecision] = {}
        self.latest: TurnDecision | None = None

    def put(self, agent: Any, td: TurnDecision) -> None:
        with self._lock:
            self._by_agent[id(agent)] = td
            self.latest = td

    def get(self, agent: Any) -> TurnDecision | None:
        with self._lock:
            return self._by_agent.get(id(agent)) if agent is not None else self.latest

    def pop(self, agent: Any) -> TurnDecision | None:
        with self._lock:
            return self._by_agent.pop(id(agent), None)


def latest_user_text(messages: list[dict[str, Any]] | None) -> str:
    for message in reversed(messages or []):
        if message.get("role") == "user":
            texts = [b["text"] for b in message.get("content", []) if isinstance(b, dict) and "text" in b]
            if texts:
                return " ".join(texts)
    return ""


def attach_frame(jpeg: bytes) -> Callable[[Any], None]:
    """Transform: add the frame to the latest user message (Bedrock Converse image block)."""

    def apply(event: Any) -> None:
        for message in reversed(event.messages or []):
            if message.get("role") == "user":
                message["content"] = list(message.get("content", [])) + [
                    {"image": {"format": "jpeg", "source": {"bytes": jpeg}}}
                ]
                return

    return apply


class SafetyFirstHandler(InterventionHandler):
    """Decider decides before any model is called; code speaks every refusal."""

    def __init__(
        self,
        client: DeciderClient,
        cache: DecisionCache,
        *,
        name: str = "which-brain-safety-first",
        thresholds: Mapping[str, float] | None = None,
        text_timeout: float = TEXT_TIMEOUT_S,
        vision_timeout: float = VISION_TIMEOUT_S,
        on_decision: Callable[[TurnDecision], None] | None = None,
    ) -> None:
        self._name = name
        self._client = client
        self._cache = cache
        self._questions = load_questions(QUESTIONS)
        self._frame_questions = load_questions(FRAME_QUESTIONS)
        self._thresholds = dict(thresholds or {})
        self._text_timeout = text_timeout
        self._vision_timeout = vision_timeout
        self._on_decision = on_decision

    @property
    def name(self) -> str:
        return self._name

    @property
    def on_error(self) -> str:
        return "deny"  # fail closed: a handler exception cancels the turn

    # The registry hooks only methods overridden at class level, so all five are here.
    async def before_invocation(self, event: Any, **kwargs: Any) -> InterventionAction:
        turn = event.invocation_state.get(TURN_KEY) or Turn(latest_user_text(event.messages))
        td = await self.decide_turn(turn)
        event.invocation_state[DECISION_KEY] = td
        self._cache.put(event.agent, td)
        if self._on_decision is not None:
            self._on_decision(td)
        if td.decision.action != "answer" or td.speak is not None:
            return Deny(reason=td.speak or "silent")
        if td.frame_task is not None and turn.frame is not None:
            return Transform(apply=attach_frame(frames.for_bedrock(turn.frame)), reason="frame attached")
        return Proceed()

    async def before_tool_call(self, event: Any, **kwargs: Any) -> InterventionAction:
        return Proceed()

    async def after_tool_call(self, event: Any, **kwargs: Any) -> InterventionAction:
        return Proceed()

    async def before_model_call(self, event: Any, **kwargs: Any) -> InterventionAction:
        return Proceed()

    async def after_model_call(self, event: Any, **kwargs: Any) -> InterventionAction:
        return Proceed()

    # ------------------------------------------------------------------ the decision

    async def decide_turn(self, turn: Turn) -> TurnDecision:
        state = rules_decide.text_state(turn.latest, turn.previous, turn.near_mic)
        try:
            resp = await asyncio.wait_for(
                self._client.aask(state, self._questions), self._text_timeout
            )
        except (DeciderUnavailable, TimeoutError):
            td = TurnDecision(rules_decide.degraded(turn.latest))
        else:
            a = resp.answers
            td = TurnDecision(
                rules_decide.decide(
                    a["addressed_to_me"], a["about_medicine"], a["brain"], **self._thresholds
                ),
                text_client_ms=resp.client_ms,
                text_server_ms=resp.server_ms,
            )
        td.decided_at = time.perf_counter()
        d = td.decision
        if d.action == "silent":
            return td
        if d.action == "ask":
            td.speak = rules.DID_YOU_ASK
            return td
        if d.action == "refuse":
            td.speak = rules.MEDICINE_REFUSE
            return td
        if d.brain in IMAGE_BRAINS and (turn.frame is not None or turn.has_camera):
            td.gate = frames.check(turn.frame)
            if not td.gate.ok:
                td.speak = rules.HOLD_UP
                return td
            td.frame_task = asyncio.ensure_future(self.check_frame(turn.frame, turn.reasks))
        return td

    async def check_frame(self, img: Image.Image, reasks: int = 0) -> FrameResult:
        try:
            resp = await asyncio.wait_for(
                self._client.aask(
                    rules_decide.FRAME_STATE, self._frame_questions, [frames.for_decider(img)]
                ),
                self._vision_timeout,
            )
        except (DeciderUnavailable, TimeoutError):
            return FrameResult("degraded")
        fc, med = resp.answers["frame_check"], resp.answers["item_is_medicine"]
        return FrameResult(
            rules_decide.frame_verdict(fc, med, reasks=reasks),
            fc,
            med,
            resp.client_ms,
            resp.server_ms,
        )


class DeciderRoutingStrategy:
    """ModelRouter strategy: the brain Decider picked, then escalate up the ladder on failure."""

    def __init__(self, cache: DecisionCache) -> None:
        self._cache = cache
        self._fallback = FallbackStrategy()

    async def select(self, context: RoutingContext, **kwargs: Any) -> RoutingCandidate | None:
        by_name = {c.name: c for c in context.candidates}
        if context.attempts:
            failed = context.attempts[-1].candidate
            order = [c.name for c in context.candidates]
            above = [c for c in context.candidates if order.index(c.name) > order.index(failed.name)]
            return await self._fallback.select(replace(context, candidates=above)) if above else None
        td = context.invocation_state.get(DECISION_KEY) or self._cache.get(
            context.invocation_state.get("agent")
        )
        brain = td.decision.brain if td is not None else None
        name = brain if brain in by_name else STRONGEST
        if td is not None:
            td.routed = name
        return by_name.get(name, context.candidates[-1])


def brain_candidates(micro: Any, lite: Any, pro: Any) -> list[RoutingCandidate]:
    """The three brains, named and described with the ``brain`` option wording from questions.yaml."""
    criteria = load_questions(QUESTIONS)["brain"].criteria or {}
    return [
        RoutingCandidate(micro, name="quiet_answer", description=criteria.get("quiet_answer")),
        RoutingCandidate(lite, name="look", description=criteria.get("look")),
        RoutingCandidate(pro, name="read_closely", description=criteria.get("read_closely")),
    ]
