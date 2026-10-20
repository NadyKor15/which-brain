"""Earcons and the timing plan: the cook always knows what to wait for.

Earcons are plain sine tones made in code (no music). The timing plan says
what is heard when: a sound within 300 ms, "Looking…" after 1 s, and never
more than 4 s of silence while the answer is on its way.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RATE = 22050

EARCONS = {
    "heard": [(880, 0.07)],  # tick: heard you
    "looking": [(660, 0.09), (880, 0.12)],  # two rising notes: hold still, looking
    "reading": [(660, 0.08), (784, 0.08), (988, 0.12)],  # three notes: reading, keep holding
    "ring": [(740, 0.18), (0, 0.08), (740, 0.18)],  # calling your helper (simulated)
}

FIRST_SOUND_S = 0.3
LOOKING_AFTER_S = 1.0
MAX_SILENCE_S = 4.0


def tone(notes: list[tuple[int, float]], rate: int = RATE) -> np.ndarray:
    parts = []
    for freq, dur in notes:
        n = int(rate * dur)
        t = np.arange(n) / rate
        wave_ = np.zeros(n) if freq == 0 else np.sin(2 * np.pi * freq * t)
        env = np.minimum(1.0, np.minimum(t, dur - t) / 0.01) if n else wave_
        parts.append(0.4 * wave_ * env)
    return np.concatenate(parts) if parts else np.zeros(0)


def write_wav(path: str | Path, samples: np.ndarray, rate: int = RATE) -> Path:
    path = Path(path)
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def write_all(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    return [write_wav(folder / f"{name}.wav", tone(notes)) for name, notes in EARCONS.items()]


@dataclass(frozen=True)
class Cue:
    at: float  # seconds after the end of the question
    what: str  # earcon name or spoken text


def plan(decided_at: float, answer_at: float, earcon: str = "looking") -> list[Cue]:
    """What is heard between the end of the question and the answer.

    decided_at: when Decider picked the brain; answer_at: when the reply is ready.
    """
    cues = [Cue(min(decided_at, FIRST_SOUND_S), earcon if decided_at <= FIRST_SOUND_S else "heard")]
    if decided_at > FIRST_SOUND_S and decided_at < answer_at:
        cues.append(Cue(decided_at, earcon))
    if answer_at > LOOKING_AFTER_S:
        cues.append(Cue(LOOKING_AFTER_S, "Looking…"))
        t = LOOKING_AFTER_S + MAX_SILENCE_S - 0.5
        while t < answer_at:
            cues.append(Cue(t, "Still looking…"))
            t += MAX_SILENCE_S - 0.5
    cues.append(Cue(answer_at, "answer"))
    return sorted(cues, key=lambda c: c.at)


def longest_silence(cues: list[Cue]) -> float:
    times = [0.0] + [c.at for c in cues]
    return max(b - a for a, b in zip(times, times[1:]))
