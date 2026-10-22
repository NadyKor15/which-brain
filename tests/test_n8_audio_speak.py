"""audio.py and speak.py without AWS: synthetic WAVs, a fake Transcribe stream and a fake Polly client."""

from __future__ import annotations

import asyncio
import io
import json

import numpy as np
import pytest

from n8 import audio, speak
from n8.spend import SpendCap, SpendCapExceeded


def tone_wav(path, amp=0.3, speech_s=1.0, lead_s=0.3, tail_s=0.6, rate=16000):
    t = np.arange(int(rate * speech_s)) / rate
    x = np.concatenate([np.zeros(int(rate * lead_s)), amp * np.sin(2 * np.pi * 220 * t), np.zeros(int(rate * tail_s))])
    speak.write_wav(path, audio.to_pcm16(x), rate)
    return path


def test_speech_level_end_of_speech_and_near_mic(tmp_path):
    loud = tone_wav(tmp_path / "cook.wav", amp=0.3)
    quiet = tone_wav(tmp_path / "child.wav", amp=0.03)
    xs, rate = audio.read_wav(loud)
    assert rate == 16000
    assert audio.end_of_speech(xs, rate) == pytest.approx(1.3, abs=0.05)
    cook = audio.speech_dbfs(xs, rate)
    child = audio.speech_dbfs(*audio.read_wav(quiet))
    assert cook - child == pytest.approx(20, abs=0.5)
    level = audio.calibrate([cook, cook - 1, cook + 1])
    assert audio.near_mic(cook, level) and not audio.near_mic(child, level)


def test_resample_keeps_duration():
    x = np.sin(np.arange(44100) / 10)
    assert audio.resample(x, 44100).size == 16000


def test_transcribe_file_charges_first_caches_and_never_streams_twice(tmp_path):
    wav = tone_wav(tmp_path / "p041.wav")
    cap = SpendCap(tmp_path / "spend.jsonl")
    cache = audio.TranscriptCache(tmp_path / "asr")
    calls = []

    async def fake(pcm, rate, eos):
        calls.append((len(pcm), rate, eos))
        assert cap.spent() > 0  # charged before the stream starts
        return audio.Heard("set a timer for twelve minutes", 420.0, 300.0, 1)

    t1 = asyncio.run(audio.transcribe_file("p041", wav, cap=cap, cache=cache, stream=fake))
    t2 = asyncio.run(audio.transcribe_file("p041", wav, cap=cap, cache=cache, stream=fake))
    assert len(calls) == 1 and t1 == t2
    assert t1.text == "set a timer for twelve minutes" and t1.final_after_eos_ms == 420.0
    assert t1.usd_ceiling == pytest.approx(15 * 0.0004)  # 15 s minimum, worst-case rate
    assert len((tmp_path / "spend.jsonl").read_text().splitlines()) == 1


def test_transcribe_over_the_ceiling_sends_nothing(tmp_path):
    wav = tone_wav(tmp_path / "p001.wav")
    cap = SpendCap(tmp_path / "spend.jsonl", max_usd=0.001)

    async def fake(*a):
        raise AssertionError("must not stream")

    with pytest.raises(SpendCapExceeded):
        asyncio.run(audio.transcribe_file("p001", wav, cap=cap, cache=audio.TranscriptCache(tmp_path), stream=fake))


class FakeStream:
    def __init__(self, data: bytes) -> None:
        self.buf = io.BytesIO(data)

    def iter_chunks(self, chunk_size=1024):
        while chunk := self.buf.read(chunk_size):
            yield chunk


class FakePolly:
    def __init__(self) -> None:
        self.calls = []

    def synthesize_speech(self, **kw):
        self.calls.append(kw)
        return {"AudioStream": FakeStream(b"\x10\x00" * 8000)}


def test_polly_charges_measures_first_byte_and_caches(tmp_path):
    cap = SpendCap(tmp_path / "spend.jsonl")
    fake = FakePolly()
    polly = speak.Polly(cap, client=fake, cache_dir=tmp_path / "tts")
    s1 = polly.say("Tomato soup, 400 grams.")
    s2 = polly.say("Tomato soup, 400 grams.")
    assert len(fake.calls) == 1 and s2.cached and s1.pcm == s2.pcm
    assert fake.calls[0]["Engine"] == "neural" and fake.calls[0]["OutputFormat"] == "pcm"
    assert s1.first_byte_ms is not None and s1.first_byte_ms <= s1.total_ms
    row = json.loads((tmp_path / "spend.jsonl").read_text())
    assert row["usd"] == pytest.approx(len("Tomato soup, 400 grams.") * 16 / 1e6, abs=1e-6)


def test_tv_line_is_quieter_than_the_cook_and_padded(tmp_path):
    polly = speak.Polly(SpendCap(tmp_path / "s.jsonl"), client=FakePolly())
    path = speak.tv_line(polly, "And now, the weather.", tmp_path / "p008.wav", turn_down_db=18)
    xs, rate = audio.read_wav(path)
    assert rate == 16000 and xs.size > 16000
    assert np.abs(xs).max() == pytest.approx(16 / 32768 * 10 ** (-18 / 20), rel=0.1)
