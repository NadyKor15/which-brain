"""Code gate for camera frames, before any model sees them.

Decider without an image still answers with high confidence, so a missing,
dark or blurred frame is caught here and the helper says "Hold it up to the
camera." without calling Bedrock. Thresholds are starting values; the final
ones come from the labelled frame set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

DARK_MEAN = 40.0  # mean luminance 0-255; start value
BLUR_VAR = 60.0  # variance of the Laplacian; start value
DECIDER_MAX_SIDE = 448

_LAPLACIAN = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


@dataclass(frozen=True)
class FrameCheck:
    status: str  # "ok", "missing", "dark" or "blurred"
    mean: float = 0.0
    sharpness: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def _gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float64)


def sharpness(img: Image.Image) -> float:
    """Variance of the Laplacian: low means blurred."""
    g = _gray(img)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (
        _LAPLACIAN[0, 1] * g[:-2, 1:-1]
        + _LAPLACIAN[1, 0] * g[1:-1, :-2]
        + _LAPLACIAN[1, 1] * g[1:-1, 1:-1]
        + _LAPLACIAN[1, 2] * g[1:-1, 2:]
        + _LAPLACIAN[2, 1] * g[2:, 1:-1]
    )
    return float(lap.var())


def check(img: Image.Image | None, dark_mean: float = DARK_MEAN, blur_var: float = BLUR_VAR) -> FrameCheck:
    if img is None:
        return FrameCheck("missing")
    small = resize_for_decider(img)
    mean = float(_gray(small).mean())
    sharp = sharpness(small)
    if mean < dark_mean:
        return FrameCheck("dark", mean, sharp)
    if sharp < blur_var:
        return FrameCheck("blurred", mean, sharp)
    return FrameCheck("ok", mean, sharp)


def load(path: str | Path | None) -> Image.Image | None:
    if path is None or not Path(path).exists():
        return None
    return Image.open(path)


def resize_for_decider(img: Image.Image, max_side: int = DECIDER_MAX_SIDE) -> Image.Image:
    w, h = img.size
    scale = max_side / max(w, h)
    if scale >= 1:
        return img.copy()
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)


def sharpest(frames: list[Image.Image]) -> int | None:
    """Index of the sharpest frame (auto-capture), or None if all fail the gate."""
    best, best_i = -1.0, None
    for i, f in enumerate(frames):
        c = check(f)
        if c.ok and c.sharpness > best:
            best, best_i = c.sharpness, i
    return best_i
