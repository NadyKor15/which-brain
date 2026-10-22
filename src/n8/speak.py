"""Speech out: Amazon Polly (neural) with the time to the first audio byte.

Earcons are made in code (earcons.py); this file only does the spoken words.
Every Polly call is priced and charged to the spend cap before it is made, and
the same text with the same voice is synthesized once (cached by sha256).

The radio/TV lines in the eval (p008-p012) are also Polly: a different voice,
turned down in code to sound like a speaker across the room. They are synthetic
and labelled so; nothing was played in a real kitchen.
"""

from __future__ import annotations

import hashlib
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .spend import SpendCap, polly_usd

VOICE = "Joanna"  # the helper
TV_VOICE = "Matthew"  # radio/TV lines in the eval
ENGINE = "neural"
RATE = 16000
REGION = "us-east-1"


@dataclass(frozen=True)
class Speech:
    text: str
    voice: str
    pcm: bytes
    first_byte_ms: float | None
    total_ms: float | None
    chars: int
    usd_ceiling: float
    cached: bool = False


def _key(text: str, voice: str) -> str:
    return hashlib.sha256(f"{ENGINE}|{voice}|{RATE}|{text}".encode()).hexdigest()[:16]


class Polly:
    """Thin wrapper: ``client`` is a boto3 Polly client (or a fake with ``synthesize_speech``)."""

    def __init__(
        self, cap: SpendCap, *, client: Any = None, cache_dir: str | Path | None = None
    ) -> None:
        self.cap = cap
        self._client = client
        self.cache_dir = Path(cache_dir) if cache_dir else None

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("polly", region_name=REGION)
        return self._client

    def say(self, text: str, voice: str = VOICE) -> Speech:
        path = self.cache_dir / f"{_key(text, voice)}.pcm" if self.cache_dir else None
        if path is not None and path.exists():
            return Speech(text, voice, path.read_bytes(), None, None, len(text), 0.0, cached=True)
        usd = polly_usd(len(text))
        self.cap.charge(usd, f"polly {voice} {len(text)} chars")
        started = time.perf_counter()
        resp = self.client.synthesize_speech(
            Engine=ENGINE, OutputFormat="pcm", SampleRate=str(RATE), Text=text, VoiceId=voice
        )
        chunks, first = [], None
        for chunk in resp["AudioStream"].iter_chunks(chunk_size=1024):
            if first is None:
                first = time.perf_counter()
            chunks.append(chunk)
        done = time.perf_counter()
        pcm = b"".join(chunks)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(pcm)
        return Speech(
            text,
            voice,
            pcm,
            None if first is None else round((first - started) * 1000, 1),
            round((done - started) * 1000, 1),
            len(text),
            round(usd, 6),
        )


def write_wav(path: str | Path, pcm: bytes, rate: int = RATE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return path


def attenuate(pcm: bytes, db: float) -> bytes:
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64) * 10 ** (-db / 20)
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()


def pad(pcm: bytes, lead_s: float = 0.3, tail_s: float = 0.6, rate: int = RATE) -> bytes:
    """Silence around a synthetic line, like the cut recorded phrases have."""
    return bytes(2 * int(rate * lead_s)) + pcm + bytes(2 * int(rate * tail_s))


def tv_line(polly: Polly, text: str, path: str | Path, turn_down_db: float = 18.0) -> Path:
    """A synthetic radio/TV line for the eval, quieter than the cook (labelled synthetic)."""
    speech = polly.say(text, TV_VOICE)
    return write_wav(path, pad(attenuate(speech.pcm, turn_down_db)))
