"""Speech in: Amazon Transcribe streaming over a recorded WAV, and the loudness flag ``near_mic``.

The eval runs Decider on what Transcribe heard, not on the script, so every
recorded phrase is streamed once and the result is cached per file (keyed by
the file's sha256). The WAV is sent in 100 ms chunks at real-time pace, so the
time from the end of speech to the final transcript is the time a cook would wait.

``near_mic`` comes from loudness: the phone is on the cook's chest, so the cook
is loud and a child across the room, another adult or the TV is quieter. It is
set against the cook's own level, measured on the tuning split only. The cook
talking to someone else is just as loud, so near_mic cannot catch that case.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import wave
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .spend import SpendCap, transcribe_usd

RATE = 16000
CHUNK_S = 0.1
FRAME_S = 0.03
REGION = "us-east-1"
LANGUAGE = "en-US"
NEAR_MIC_MARGIN_DB = 6.0  # start value; set on the tuning split


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Mono float samples in [-1, 1] and the sample rate (16-bit PCM WAV)."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {8 * w.getsampwidth()}-bit")
        rate, channels = w.getframerate(), w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64)
    if channels > 1:
        raw = raw.reshape(-1, channels).mean(axis=1)
    return raw / 32768.0, rate


def resample(samples: np.ndarray, rate: int, target: int = RATE) -> np.ndarray:
    if rate == target or samples.size == 0:
        return samples
    n = max(1, round(samples.size * target / rate))
    x_old = np.arange(samples.size) / rate
    return np.interp(np.arange(n) / target, x_old, samples)


def to_pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


def frame_dbfs(samples: np.ndarray, rate: int, frame_s: float = FRAME_S) -> np.ndarray:
    n = max(1, int(rate * frame_s))
    usable = samples[: samples.size // n * n]
    if usable.size == 0:
        return np.array([-120.0])
    rms = np.sqrt(np.mean(usable.reshape(-1, n) ** 2, axis=1))
    return 20 * np.log10(np.maximum(rms, 1e-6))


def speech_dbfs(samples: np.ndarray, rate: int, window_db: float = 20.0) -> float:
    """Level of the speech itself: RMS over the frames within ``window_db`` of the loudest one."""
    db = frame_dbfs(samples, rate)
    active = db[db >= db.max() - window_db]
    power = np.mean(10 ** (active / 10))
    return float(10 * np.log10(power))


def end_of_speech(samples: np.ndarray, rate: int, below_peak_db: float = 30.0) -> float:
    """Seconds from the start of the file to the end of the last frame of speech."""
    db = frame_dbfs(samples, rate)
    loud = np.nonzero(db >= max(db.max() - below_peak_db, -60.0))[0]
    if loud.size == 0:
        return samples.size / rate
    return float((loud[-1] + 1) * FRAME_S)


def calibrate(cook_levels: Iterable[float]) -> float:
    """The cook's own speech level: the median over the tuning split's cook phrases."""
    levels = sorted(cook_levels)
    if not levels:
        raise ValueError("no cook phrases to calibrate on")
    return float(np.median(levels))


def near_mic(level_dbfs: float, cook_dbfs: float, margin_db: float = NEAR_MIC_MARGIN_DB) -> bool:
    return level_dbfs >= cook_dbfs - margin_db


@dataclass(frozen=True)
class Heard:
    """What one streaming pass returned, with times relative to the end of speech."""

    text: str
    final_after_eos_ms: float | None
    first_partial_ms: float | None
    segments: int


@dataclass(frozen=True)
class Transcript:
    id: str
    sha256: str
    text: str
    audio_s: float
    end_of_speech_s: float
    speech_dbfs: float
    final_after_eos_ms: float | None
    first_partial_ms: float | None
    segments: int
    usd_ceiling: float
    language: str = LANGUAGE


StreamFn = Callable[[bytes, int, float], Awaitable[Heard]]
"""(pcm16, rate, end_of_speech_s) -> Heard. The real one is :func:`transcribe_stream`."""


async def transcribe_stream(
    pcm: bytes,
    rate: int,
    end_of_speech_s: float,
    *,
    region: str = REGION,
    language: str = LANGUAGE,
    realtime: bool = True,
) -> Heard:
    """Stream PCM to Amazon Transcribe and collect the final segments (paid)."""
    from amazon_transcribe.client import TranscribeStreamingClient
    from amazon_transcribe.handlers import TranscriptResultStreamHandler

    client = TranscribeStreamingClient(region=region)
    stream = await client.start_stream_transcription(
        language_code=language, media_sample_rate_hz=rate, media_encoding="pcm"
    )
    started = time.perf_counter()
    eos_wall = started + end_of_speech_s
    finals: list[str] = []
    marks: dict[str, float] = {}

    class Collect(TranscriptResultStreamHandler):
        async def handle_transcript_event(self, event):  # noqa: ANN001 - SDK type
            now = time.perf_counter()
            for result in event.transcript.results:
                if not result.alternatives:
                    continue
                marks.setdefault("first_partial", now)
                if not result.is_partial:
                    finals.append(result.alternatives[0].transcript)
                    marks["final"] = now

    async def send() -> None:
        step = int(rate * CHUNK_S) * 2
        for i, offset in enumerate(range(0, len(pcm), step)):
            if realtime:
                delay = started + i * CHUNK_S - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            await stream.input_stream.send_audio_event(audio_chunk=pcm[offset : offset + step])
        await stream.input_stream.end_stream()

    await asyncio.gather(send(), Collect(stream.output_stream).handle_events())
    final = marks.get("final")
    partial = marks.get("first_partial")
    return Heard(
        text=" ".join(t.strip() for t in finals if t.strip()),
        final_after_eos_ms=None if final is None else (final - eos_wall) * 1000,
        first_partial_ms=None if partial is None else (partial - started) * 1000,
        segments=len(finals),
    )


class TranscriptCache:
    """One JSON file per recorded phrase; a changed WAV gets a new key, never a stale text."""

    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)

    def _path(self, phrase_id: str, sha: str) -> Path:
        return self.folder / f"{phrase_id}-{sha[:12]}.json"

    def get(self, phrase_id: str, sha: str) -> Transcript | None:
        path = self._path(phrase_id, sha)
        return Transcript(**json.loads(path.read_text())) if path.exists() else None

    def put(self, t: Transcript) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self._path(t.id, t.sha256).write_text(json.dumps(asdict(t), indent=1))


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


async def transcribe_file(
    phrase_id: str,
    path: str | Path,
    *,
    cap: SpendCap,
    cache: TranscriptCache,
    stream: StreamFn | None = None,
) -> Transcript:
    """Cached text for one recorded phrase; streams it (and charges the cap first) if new."""
    sha = file_sha256(path)
    hit = cache.get(phrase_id, sha)
    if hit is not None:
        return hit
    samples, rate = read_wav(path)
    samples = resample(samples, rate)
    audio_s = samples.size / RATE
    eos = end_of_speech(samples, RATE)
    usd = transcribe_usd(audio_s)
    cap.charge(usd, f"transcribe {phrase_id} {audio_s:.1f}s")
    heard = await (stream or transcribe_stream)(to_pcm16(samples), RATE, eos)
    t = Transcript(
        id=phrase_id,
        sha256=sha,
        text=heard.text,
        audio_s=round(audio_s, 3),
        end_of_speech_s=round(eos, 3),
        speech_dbfs=round(speech_dbfs(samples, RATE), 2),
        final_after_eos_ms=None
        if heard.final_after_eos_ms is None
        else round(heard.final_after_eos_ms, 1),
        first_partial_ms=None
        if heard.first_partial_ms is None
        else round(heard.first_partial_ms, 1),
        segments=heard.segments,
        usd_ceiling=round(usd, 6),
    )
    cache.put(t)
    return t
