"""say_how_sure against a fake Decider server: the image gate, the hedge, the re-ask cap, medicine."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from n8 import rules

from .conftest import FakeServer, choice, noul, ok_response

HERE = Path(__file__).resolve().parents[1] / "patterns"


def load():
    spec = importlib.util.spec_from_file_location(
        "say_how_sure_handler", HERE / "say_how_sure" / "handler.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


shs = load()


def sharp() -> Image.Image:
    return Image.fromarray(np.random.default_rng(1).integers(0, 255, (300, 400, 3), dtype=np.uint8))


def server(probs, item_medicine=0.02):
    top = max(probs, key=probs.get)
    return FakeServer(
        lambda body: ok_response(
            {"frame_check": choice(top, probs, probs[top]), "item_is_medicine": noul(item_medicine)}
        )
    )


def test_gate_stops_no_frame_black_and_blur_before_any_model():
    assert shs.image_gate(None).status == "missing"
    assert shs.image_gate(Image.new("RGB", (400, 300), (3, 3, 3))).status == "dark"
    assert shs.image_gate(Image.new("RGB", (400, 300), (128, 128, 128))).status == "blurred"
    assert shs.image_gate(sharp()).ok
    assert shs.speak(shs.image_gate(None), None, "Tomato soup.") == rules.HOLD_UP


def test_clear_is_plain_partly_is_hedged_and_numbers_are_never_hedged():
    clear = asyncio.run(
        shs.look(server({"clear": 0.95, "partly": 0.04, "unreadable": 0.01}).client(), sharp())
    )
    assert clear.verdict == "plain" and shs.hedge(clear.verdict, "Tomato soup.") == "Tomato soup."
    partly = asyncio.run(
        shs.look(server({"clear": 0.3, "partly": 0.6, "unreadable": 0.1}).client(), sharp())
    )
    assert partly.verdict == "hedge"
    assert shs.hedge("hedge", "A red can.").startswith("I think")
    said = shs.hedge("hedge", "Bake 45 minutes at 350 degrees.")
    assert not rules.hedge_violations(said) and "cut off" in said


def test_unreadable_reasks_at_most_twice():
    bad = {"clear": 0.05, "partly": 0.15, "unreadable": 0.8}
    loop = shs.ReaskLoop()
    seen = []
    for _ in range(4):
        r = asyncio.run(shs.look(server(bad).client(), sharp(), reasks=loop.count))
        seen.append(loop.next(r.verdict))
    assert seen[:3] == ["reask", "reask", "hedge"]
    assert shs.hedge("reask", "x") == rules.HOLD_CLOSER


def test_medicine_in_the_picture_refuses():
    r = asyncio.run(
        shs.look(
            server({"clear": 0.95, "partly": 0.04, "unreadable": 0.01}, 0.35).client(), sharp()
        )
    )
    assert (
        r.verdict == "refuse"
        and shs.hedge(r.verdict, "Acetaminophen 500 mg") == rules.MEDICINE_REFUSE
    )


def test_decider_down_means_hedge_never_plain():
    down = FakeServer(lambda body: (503, {"detail": "down"}))
    r = asyncio.run(shs.look(down.client(), sharp()))
    assert r.verdict == "degraded" and shs.hedge(r.verdict, "A red can.").startswith("I think")


def test_question_files_of_both_patterns_say_the_same():
    a = yaml.safe_load((HERE / "say_how_sure" / "questions.yaml").read_text())
    b = yaml.safe_load((HERE / "brain_picker" / "frame_questions.yaml").read_text())
    assert a == b
