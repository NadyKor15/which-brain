"""Hard ceiling on paid AWS calls (approved 2026-10-21: up to $5, AWS credits only).

Every paid call is priced worst-case BEFORE it is made. If the ledger total plus
this call would pass the ceiling, the call is refused and nothing is sent.
The ledger is cumulative across runs, so the ceiling covers the whole build,
not one run. Prices: eval/cost.md (AWS public price list, us-east-1, 2026-10-20).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

MAX_USD = 5.0

# USD per 1K tokens (in, out)
NOVA = {
    "micro": (0.000035, 0.00014),
    "lite": (0.00006, 0.00024),
    "pro": (0.0008, 0.0032),
}
TRANSCRIBE_PER_S = 0.0004  # worst case; list price is $0.0001667/s
TRANSCRIBE_MIN_S = 15
POLLY_NEURAL_PER_CHAR = 16 / 1_000_000


class SpendCapExceeded(RuntimeError):
    """The call would take total spend past the approved ceiling."""


def nova_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    p_in, p_out = NOVA[model]
    return tokens_in / 1000 * p_in + tokens_out / 1000 * p_out


def transcribe_usd(seconds: float) -> float:
    return max(seconds, TRANSCRIBE_MIN_S) * TRANSCRIBE_PER_S


def polly_usd(chars: int) -> float:
    return chars * POLLY_NEURAL_PER_CHAR


@dataclass
class SpendCap:
    ledger: Path
    max_usd: float = MAX_USD

    def spent(self) -> float:
        if not self.ledger.exists():
            return 0.0
        return sum(json.loads(line)["usd"] for line in self.ledger.open() if line.strip())

    def charge(self, usd: float, what: str) -> float:
        """Record a call before making it; raises instead if it would pass the ceiling."""
        total = self.spent() + usd
        if total > self.max_usd:
            raise SpendCapExceeded(
                f"{what}: ${usd:.4f} would bring spend to ${total:.4f}, ceiling ${self.max_usd:.2f}"
            )
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "what": what, "usd": round(usd, 6)}) + "\n")
        return total
