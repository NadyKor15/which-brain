"""brain_picker against a fake Decider server (no model load) and the REAL Strands Agent + ModelRouter.

Covers: OR refusal, P(yes) thresholds and the ask band, ladder escalation, the strategy returning a
candidate by identity, both paths from the handler to the router (invocation_state and the
DecisionCache), fail-closed when the server is down, the image gate, and the frame verdicts.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image
from strands.interventions import Deny, Proceed, Transform
from strands.models.model import Model
from strands.models.routing import RoutingAttempt, RoutingContext

from n8 import rules
from n8.agent import bp, build_agent, run_turn
from n8.brains import Brains, CappedModel
from n8.spend import SpendCap, SpendCapExceeded

from .conftest import FakeServer, choice, noul, ok_response

BRAIN = {"quiet_answer": 0.02, "look": 0.95, "read_closely": 0.02, "ask_human": 0.01}


class TextModel(Model):
    """Scripted model: answers with fixed text, or raises when ``fail`` is set."""

    def __init__(self, name: str, text: str = "", fail: bool = False) -> None:
        self.name, self.text, self.fail = name, text or f"answer from {name}", fail
        self.seen: list[Any] = []

    def update_config(self, **model_config: Any) -> None:
        pass

    def get_config(self) -> Any:
        return {"model_id": self.name}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError
        yield  # pragma: no cover

    async def stream(
        self, messages, tool_specs=None, system_prompt=None, **kwargs
    ) -> AsyncIterable:
        self.seen.append(messages)
        if self.fail:
            raise RuntimeError(f"{self.name} is down")
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": self.text}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield {
            "metadata": {
                "usage": {"inputTokens": 100, "outputTokens": 10, "totalTokens": 110},
                "metrics": {"latencyMs": 1},
            }
        }


def brains(tmp_path, fail: tuple[str, ...] = (), texts: dict[str, str] | None = None) -> Brains:
    cap = SpendCap(tmp_path / "spend.jsonl")
    texts = texts or {}
    return Brains(
        *(
            CappedModel(TextModel(k, texts.get(k, ""), k in fail), k, cap)
            for k in ("micro", "lite", "pro")
        )
    )


def text_server(addressed=0.95, medicine=0.02, brain=None, frame=None, item_medicine=0.02):
    brain = brain or BRAIN

    def respond(body):
        if "frame_check" in body["questions"]:
            probs = frame or {"clear": 0.95, "partly": 0.04, "unreadable": 0.01}
            top = max(probs, key=probs.get)
            return ok_response(
                {
                    "frame_check": choice(top, probs, probs[top]),
                    "item_is_medicine": noul(item_medicine),
                }
            )
        top = max(brain, key=brain.get)
        return ok_response(
            {
                "addressed_to_me": noul(addressed),
                "about_medicine": noul(medicine),
                "brain": choice(top, brain, brain[top]),
            }
        )

    return FakeServer(respond)


def label_image(sharp: bool = True) -> Image.Image:
    rng = np.random.default_rng(0)
    arr = (
        rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)
        if sharp
        else np.full((300, 400, 3), 128, np.uint8)
    )
    return Image.fromarray(arr)


def turn(text="What's in this can?", frame=None, has_camera=False, **kw):
    return bp.Turn(latest=text, frame=frame, has_camera=has_camera, **kw)


def play(tmp_path, server, t, fail=(), texts=None, strategy=None):
    b = brains(tmp_path, fail, texts)
    cache = bp.DecisionCache()
    agent = build_agent(b, server.client(), cache, strategy=strategy)
    return asyncio.run(run_turn(t, agent, cache, b)), b, server


# ---------------------------------------------------------------- the decision rules


def test_not_for_me_is_silent_and_no_model_is_called(tmp_path):
    reply, b, _ = play(tmp_path, text_server(addressed=0.05), turn())
    assert reply.action == "silent" and reply.said is None and reply.earcon is None
    assert all(not m.inner.seen for m in b.all())


def test_ask_band_says_did_you_ask_once_without_a_model_call(tmp_path):
    reply, b, _ = play(tmp_path, text_server(addressed=0.5), turn())
    assert reply.action == "ask" and reply.said == rules.DID_YOU_ASK
    assert all(not m.inner.seen for m in b.all())


def test_medicine_refuses_on_p_yes_well_below_half(tmp_path):
    reply, b, _ = play(tmp_path, text_server(medicine=0.31), turn("Is this the one for my heart?"))
    assert reply.action == "refuse" and reply.said == rules.MEDICINE_REFUSE
    assert all(not m.inner.seen for m in b.all())
    reply, _, _ = play(
        tmp_path,
        text_server(
            medicine=0.29,
            brain={"quiet_answer": 0.95, "look": 0.02, "read_closely": 0.02, "ask_human": 0.01},
        ),
        turn("How many ounces in a pound?"),
    )
    assert reply.action == "answer"


def test_ask_human_top_choice_refuses_at_any_confidence(tmp_path):
    brain = {"quiet_answer": 0.30, "look": 0.20, "read_closely": 0.15, "ask_human": 0.35}
    reply, _, _ = play(tmp_path, text_server(brain=brain), turn())
    assert reply.action == "refuse"


def test_item_medicine_in_the_frame_refuses_after_the_answer_is_discarded(tmp_path):
    reply, b, _ = play(
        tmp_path,
        text_server(item_medicine=0.4),
        turn(frame=label_image()),
        texts={"lite": "This is Acetaminophen 500 mg."},
    )
    assert reply.action == "refuse" and reply.said == rules.MEDICINE_REFUSE
    assert "500" not in reply.said


# ---------------------------------------------------------------- routing


def test_router_runs_the_brain_decider_picked(tmp_path):
    for brain, model in (("quiet_answer", "micro"), ("look", "lite"), ("read_closely", "pro")):
        probs = {k: (0.94 if k == brain else 0.02) for k in BRAIN}
        reply, b, _ = play(tmp_path, text_server(brain=probs), turn("Set a timer for ten minutes."))
        assert reply.routed == brain and reply.model == model
        ran = [m.model for m in b.all() if m.inner.seen]
        assert ran == [model]


def test_unsure_brain_takes_the_stronger_of_the_top_two(tmp_path):
    probs = {"quiet_answer": 0.55, "look": 0.05, "read_closely": 0.40, "ask_human": 0.0}
    reply, _, _ = play(tmp_path, text_server(brain=probs), turn("How long do I bake this?"))
    assert reply.routed == "read_closely"


def test_failed_brain_escalates_up_the_ladder(tmp_path):
    probs = {k: (0.94 if k == "quiet_answer" else 0.02) for k in BRAIN}
    reply, b, _ = play(tmp_path, text_server(brain=probs), turn("Set a timer."), fail=("micro",))
    assert reply.action == "answer" and reply.answer == "answer from lite"
    assert [m.model for m in b.all() if m.inner.seen] == ["micro", "lite"]


def test_top_brain_failing_ends_routing_and_speaks_sorry(tmp_path):
    probs = {k: (0.94 if k == "read_closely" else 0.02) for k in BRAIN}
    reply, _, _ = play(
        tmp_path, text_server(brain=probs), turn("Read the second step."), fail=("pro",)
    )
    assert reply.action == "error" and reply.said == bp.SORRY


def _context(candidates, state, attempts=()):
    return RoutingContext(
        messages=[],
        system_prompt=None,
        tool_specs=[],
        candidates=candidates,
        invocation_state=state,
        attempts=attempts,
    )


def test_strategy_returns_the_candidate_instance_by_identity_on_both_paths(tmp_path):
    b = brains(tmp_path)
    cands = tuple(bp.brain_candidates(b.micro, b.lite, b.pro))
    cache = bp.DecisionCache()
    strategy = bp.DeciderRoutingStrategy(cache)
    from n8.decide import Decision

    td = bp.TurnDecision(Decision("answer", "look"))
    # path 1: invocation_state carries the decision
    got = asyncio.run(strategy.select(_context(cands, {bp.DECISION_KEY: td})))
    assert got is cands[1]
    # path 2: invocation_state is empty (as on the agent.py:1086 path); the cache answers
    agent = object()
    cache.put(agent, bp.TurnDecision(Decision("answer", "quiet_answer")))
    assert asyncio.run(strategy.select(_context(cands, {"agent": agent}))) is cands[0]
    # nothing known: the strongest brain
    assert (
        asyncio.run(bp.DeciderRoutingStrategy(bp.DecisionCache()).select(_context(cands, {})))
        is cands[2]
    )
    # after lite failed: up, never down
    up = asyncio.run(
        strategy.select(_context(cands, {}, (RoutingAttempt(cands[1], RuntimeError()),)))
    )
    assert up is cands[2]


def test_invocation_state_reaches_the_router_in_the_real_agent(tmp_path):
    seen = []

    class Spy(bp.DeciderRoutingStrategy):
        async def select(self, context, **kwargs):
            seen.append(bp.DECISION_KEY in context.invocation_state)
            return await super().select(context, **kwargs)

    cache = bp.DecisionCache()
    play(tmp_path, text_server(), turn(), strategy=Spy(cache))
    assert seen == [True]


# ---------------------------------------------------------------- fail closed


def test_decider_down_uses_the_keyword_net_and_the_strong_brain(tmp_path):
    down = FakeServer(lambda body: (503, {"detail": "down"}))
    reply, _, _ = play(tmp_path, down, turn("Is this my blood pressure pill?"))
    assert reply.action == "refuse" and reply.degraded and reply.said == rules.MEDICINE_REFUSE
    reply, b, _ = play(tmp_path, down, turn("How many ounces in a pound?"))
    assert reply.action == "answer" and reply.degraded and reply.routed == "read_closely"


def test_slow_decider_times_out_into_degraded_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "TEXT_TIMEOUT_S", 0.05)
    server = text_server()

    class Slow:
        async def aask(self, *a, **k):
            await asyncio.sleep(1)

    b = brains(tmp_path)
    cache = bp.DecisionCache()
    agent = build_agent(b, server.client(), cache)
    handler = (
        agent._intervention_registry._handlers[0]
        if hasattr(agent, "_intervention_registry")
        else None
    )
    h = bp.SafetyFirstHandler(Slow(), cache, text_timeout=0.05)
    started = time.perf_counter()
    td = asyncio.run(h.decide_turn(turn("Is it time for my insulin?")))
    assert td.decision.degraded and td.decision.action == "refuse"
    assert time.perf_counter() - started < 0.5
    assert handler is None or handler.on_error == "deny"


def test_handler_exception_is_denied_and_code_says_sorry(tmp_path):
    class Boom:
        async def aask(self, *a, **k):
            raise ValueError("bug")

    b = brains(tmp_path)
    cache = bp.DecisionCache()
    agent = build_agent(b, Boom(), cache)
    reply = asyncio.run(run_turn(turn(), agent, cache, b))
    assert reply.action == "error" and reply.said == bp.SORRY
    assert all(not m.inner.seen for m in b.all())


# ---------------------------------------------------------------- the frame


def test_image_gate_stops_missing_dark_and_blurred_frames_before_bedrock(tmp_path):
    for frame in (None, Image.new("RGB", (400, 300), (5, 5, 5)), label_image(sharp=False)):
        reply, b, server = play(tmp_path, text_server(), turn(frame=frame, has_camera=True))
        assert reply.said == rules.HOLD_UP and reply.verdict.startswith("gate:")
        assert all(not m.inner.seen for m in b.all())
        assert not any("frame_check" in r["questions"] for r in server.requests)


def test_look_attaches_the_frame_and_runs_request_two(tmp_path):
    reply, b, server = play(
        tmp_path, text_server(), turn(frame=label_image()), texts={"lite": "Tomato soup."}
    )
    sent = b.lite.inner.seen[0][-1]["content"]
    assert any("image" in block for block in sent)
    assert any("frame_check" in r["questions"] and r.get("images") for r in server.requests)
    assert reply.verdict == "plain" and reply.said == "Tomato soup."


def test_partly_hedges_words_but_never_a_number(tmp_path):
    partly = {"clear": 0.3, "partly": 0.6, "unreadable": 0.1}
    reply, _, _ = play(
        tmp_path,
        text_server(frame=partly),
        turn(frame=label_image()),
        texts={"lite": "Red lid, round can."},
    )
    assert reply.verdict == "hedge" and reply.said.startswith("I think")
    reply, _, _ = play(
        tmp_path,
        text_server(frame=partly),
        turn(frame=label_image()),
        texts={"lite": "Bake 45 minutes at 350 degrees."},
    )
    # no OCR at runtime: a hedged number asks for a closer look, never quotes the model as the label
    assert not rules.hedge_violations(reply.said) and reply.said == rules.HOLD_CLOSER


def test_unreadable_asks_closer_at_most_twice(tmp_path):
    bad = {"clear": 0.05, "partly": 0.15, "unreadable": 0.8}
    for reasks, want in ((0, "reask"), (1, "reask"), (2, "hedge")):
        reply, _, _ = play(
            tmp_path,
            text_server(frame=bad),
            turn(frame=label_image(), reasks=reasks),
            texts={"lite": "Tomato soup."},
        )
        assert reply.verdict == want
    assert reply.said.startswith("I think")


def test_request_two_runs_next_to_bedrock_not_after_it(tmp_path):
    """The frame check is started before the model call, so the two overlap."""
    order: list[str] = []

    class Clock(TextModel):
        async def stream(self, messages, **kw):
            order.append("model")
            async for e in super().stream(messages, **kw):
                yield e

    server = text_server()
    orig = server.handle

    def handle(request):
        if b"frame_check" in request.content:
            order.append("frame")
        return orig(request)

    server.handle = handle
    cap = SpendCap(tmp_path / "s.jsonl")
    b = Brains(*(CappedModel(Clock(k), k, cap) for k in ("micro", "lite", "pro")))
    cache = bp.DecisionCache()
    agent = build_agent(b, server.client(), cache)
    reply = asyncio.run(run_turn(turn(frame=label_image()), agent, cache, b))
    assert reply.action == "answer" and set(order) == {"frame", "model"}


# ---------------------------------------------------------------- the spend cap


def test_capped_model_refuses_a_call_past_the_ceiling_and_sends_nothing(tmp_path):
    cap = SpendCap(tmp_path / "s.jsonl", max_usd=0.00001)
    inner = TextModel("pro")
    m = CappedModel(inner, "pro", cap)

    async def go():
        async for _ in m.stream([{"role": "user", "content": [{"text": "hi" * 500}]}]):
            pass

    with pytest.raises(SpendCapExceeded):
        asyncio.run(go())
    assert inner.seen == [] and not (tmp_path / "s.jsonl").exists()


def test_capped_model_records_usage_and_first_token(tmp_path):
    b = brains(tmp_path)

    async def go():
        async for _ in b.micro.stream([{"role": "user", "content": [{"text": "hi"}]}]):
            pass

    asyncio.run(go())
    (call,) = b.take_calls()
    assert call.tokens_in == 100 and call.tokens_out == 10 and call.first_token_ms is not None
    assert 0 < call.usd < call.usd_ceiling or call.usd_ceiling > 0
    assert SpendCap(tmp_path / "spend.jsonl").spent() == pytest.approx(call.usd_ceiling, abs=1e-6)


def test_handler_overrides_all_five_lifecycle_methods_at_class_level():
    for name in (
        "before_invocation",
        "before_tool_call",
        "after_tool_call",
        "before_model_call",
        "after_model_call",
    ):
        assert name in vars(bp.SafetyFirstHandler)
    h = bp.SafetyFirstHandler(SimpleNamespace(), bp.DecisionCache())
    assert h.on_error == "deny" and h.name == "which-brain-safety-first"
    assert isinstance(Deny(reason="x"), Deny) and Proceed and Transform


def test_worst_case_price_covers_the_calls_seen_on_bedrock():
    """Regression from the 2026-10-22 smoke run: the ceiling must never be below the real price."""
    from pydantic import BaseModel, Field

    from n8.agent import SYSTEM_PROMPT
    from n8.brains import worst_case_usd
    from n8.spend import nova_usd

    class Q12(BaseModel):
        addressed_to_me: bool = Field(
            description="Is the latest utterance a request to the assistant?"
        )
        about_medicine: bool = Field(description="Would a correct answer require a medicine?")

    msgs = [
        {"role": "user", "content": [{"text": "Latest utterance: 'How many ounces in a pound?'"}]}
    ]
    spec = [{"name": "Q12", "inputSchema": {"json": Q12.model_json_schema()}}]
    est = worst_case_usd(
        "micro", msgs, "Answer about the latest utterance spoken in a kitchen.", 160, spec
    )
    assert est >= nova_usd("micro", 558, 74)  # seen: 558 in, 74 out
    img = [
        {
            "role": "user",
            "content": [{"text": "What's in this can?"}, {"image": {"format": "jpeg"}}],
        }
    ]
    assert worst_case_usd("pro", img, SYSTEM_PROMPT, 160) >= nova_usd(
        "pro", 1692, 8
    )  # seen: 1692 in


def test_near_mic_modes_state_off_and_code(tmp_path):
    server = text_server(addressed=0.95)
    cache = bp.DecisionCache()
    shown = bp.SafetyFirstHandler(server.client(), cache, near_mic="state")
    asyncio.run(shown.decide_turn(turn(near_mic=False)))
    assert '"near_mic": false' in server.requests[-1]["state"]
    off = bp.SafetyFirstHandler(server.client(), cache, near_mic="off")
    asyncio.run(off.decide_turn(turn(near_mic=False)))
    assert "near_mic" not in server.requests[-1]["state"]
    n = len(server.requests)
    code = bp.SafetyFirstHandler(server.client(), cache, near_mic="code")
    td = asyncio.run(code.decide_turn(turn(near_mic=False)))
    assert td.decision.action == "silent" and len(server.requests) == n  # no Decider call
    td = asyncio.run(code.decide_turn(turn(near_mic=True)))
    assert td.decision.action == "answer" and "near_mic" not in server.requests[-1]["state"]


# ---------------------------------------------------------------- review findings, 2026-10-22


def test_vision_check_down_with_a_pill_in_frame_is_not_answered(tmp_path):
    def respond(body):
        if "frame_check" in body["questions"]:
            return 503, {"detail": "vision down"}
        return text_server().responder(body)

    reply, _, _ = play(
        tmp_path,
        FakeServer(respond),
        turn("What's this?", frame=label_image()),
        texts={"lite": "This is a small white bottle."},
    )
    assert reply.said == bp.SORRY and reply.verdict == "degraded"


def test_vision_check_4xx_fails_closed_and_does_not_crash(tmp_path):
    def respond(body):
        if "frame_check" in body["questions"]:
            return 422, {"detail": "bad image"}
        return text_server().responder(body)

    reply, _, _ = play(
        tmp_path, FakeServer(respond), turn(frame=label_image()), texts={"lite": "Tomato soup."}
    )
    assert reply.said == bp.SORRY and reply.verdict == "degraded"


def test_quiet_brain_with_a_pill_in_view_still_refuses_and_sends_no_frame(tmp_path):
    quiet = {"quiet_answer": 0.95, "look": 0.02, "read_closely": 0.02, "ask_human": 0.01}
    reply, b, server = play(
        tmp_path,
        text_server(brain=quiet, item_medicine=0.6),
        turn("Can I have two of these with my coffee?", frame=label_image()),
        texts={"micro": "Yes, two is fine."},
    )
    assert reply.action == "refuse" and reply.said == rules.MEDICINE_REFUSE
    assert not any("image" in blk for blk in b.micro.inner.seen[0][-1]["content"])
    assert any("frame_check" in r["questions"] for r in server.requests)


def test_model_answer_about_medicine_or_a_dose_is_refused_by_code(tmp_path):
    quiet = {"quiet_answer": 0.95, "look": 0.02, "read_closely": 0.02, "ask_human": 0.01}
    for text in ("Take 2 tablets after dinner.", "That is your heart medicine."):
        reply, _, _ = play(
            tmp_path, text_server(brain=quiet), turn("Is this the one?"), texts={"micro": text}
        )
        assert reply.action == "refuse" and reply.said == rules.MEDICINE_REFUSE


def test_i_think_never_stands_next_to_a_number(tmp_path):
    from n8.agent import NOT_SURE

    quiet = {"quiet_answer": 0.95, "look": 0.02, "read_closely": 0.02, "ask_human": 0.01}
    reply, _, _ = play(
        tmp_path,
        text_server(brain=quiet),
        turn("How long do I boil an egg?"),
        texts={"micro": "I think about 9 minutes."},
    )
    assert reply.said == NOT_SURE
    reply, _, _ = play(
        tmp_path,
        text_server(),
        turn(frame=label_image()),
        texts={"lite": "I think it says bake at 180 degrees."},
    )
    assert reply.verdict == "hedged_number" and reply.said == rules.HOLD_CLOSER
    reply, _, _ = play(
        tmp_path,
        text_server(brain=quiet),
        turn("How many?"),
        texts={"micro": "I think there are sixteen in the pack."},
    )
    assert reply.said == NOT_SURE


def test_routed_names_the_brain_that_answered_after_escalation(tmp_path):
    probs = {k: (0.94 if k == "quiet_answer" else 0.02) for k in BRAIN}
    reply, _, _ = play(tmp_path, text_server(brain=probs), turn("Set a timer."), fail=("micro",))
    assert reply.routed == "look" and reply.model == "lite"


def test_frame_task_is_cancelled_when_every_brain_fails(tmp_path, monkeypatch):
    async def hang(self, img, reasks=0):
        await asyncio.sleep(30)

    monkeypatch.setattr(bp.SafetyFirstHandler, "check_frame", hang)
    seen = []
    b = brains(tmp_path, fail=("micro", "lite", "pro"))
    cache = bp.DecisionCache()
    agent = build_agent(b, text_server().client(), cache, on_decision=seen.append)

    async def go():
        reply = await run_turn(turn(frame=label_image()), agent, cache, b)
        await asyncio.sleep(0)
        return reply

    started = time.perf_counter()
    reply = asyncio.run(go())
    assert reply.said == bp.SORRY and time.perf_counter() - started < 5
    assert seen and seen[0].frame_task.cancelled()
