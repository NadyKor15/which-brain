import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from n8 import earcons, frames


def _label(text="TOMATO SOUP 400 g", size=(800, 500)):
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    for i in range(6):
        d.text((40, 40 + i * 70), text, fill="black")
    d.rectangle((20, 20, size[0] - 20, size[1] - 20), outline="black", width=6)
    return img


def test_sharp_label_passes():
    c = frames.check(_label())
    assert c.status == "ok", c


def test_missing_dark_and_blurred_are_caught():
    assert frames.check(None).status == "missing"
    assert frames.check(Image.new("RGB", (640, 480), (10, 10, 10))).status == "dark"
    blurred = _label().filter(ImageFilter.GaussianBlur(12))
    assert frames.check(blurred).status == "blurred"


def test_resize_keeps_aspect_and_max_side():
    small = frames.resize_for_decider(Image.new("RGB", (4032, 3024)))
    assert max(small.size) == 448 and small.size == (448, 336)
    tiny = Image.new("RGB", (100, 50))
    assert frames.resize_for_decider(tiny).size == (100, 50)


def test_sharpest_picks_the_sharp_frame():
    sharp = _label()
    soft = sharp.filter(ImageFilter.GaussianBlur(3))
    blurred = sharp.filter(ImageFilter.GaussianBlur(12))
    assert frames.sharpest([blurred, soft, sharp]) == 2
    assert frames.sharpest([blurred, Image.new("RGB", (64, 64))]) is None


def test_earcons_are_written(tmp_path):
    paths = earcons.write_all(tmp_path)
    assert {p.stem for p in paths} == set(earcons.EARCONS)
    with wave.open(str(paths[0])) as w:
        assert w.getframerate() == earcons.RATE and w.getnframes() > 0
    assert np.abs(earcons.tone(earcons.EARCONS["looking"])).max() <= 0.41


def test_first_sound_within_300_ms_and_never_silent_over_4_s():
    for decided, answer in [(0.1, 0.8), (0.1, 2.5), (0.6, 3.0), (0.12, 11.0), (1.4, 9.3)]:
        cues = earcons.plan(decided, answer)
        assert cues[0].at <= earcons.FIRST_SOUND_S
        assert earcons.longest_silence(cues) <= earcons.MAX_SILENCE_S, (decided, answer, cues)
        assert cues[-1].what == "answer"


def test_looking_only_when_the_answer_is_slow():
    fast = [c.what for c in earcons.plan(0.1, 0.8)]
    assert "Looking…" not in fast
    slow = [c.what for c in earcons.plan(0.1, 2.0)]
    assert "Looking…" in slow
