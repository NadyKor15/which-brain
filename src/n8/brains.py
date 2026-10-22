"""The three Bedrock brains (Amazon Nova only) and the spend cap in front of every call.

``CappedModel`` wraps a Strands model: before each call it prices the request at
its worst case (text tokens from characters, a fixed allowance per image, the
full ``max_tokens`` out) and charges the cap, so a call that would pass the
approved ceiling is never sent. After the call it keeps the real usage and the
time to the first text token. Model IDs are in-Region (no ``us.`` profile):
AWS Free Tier credits do not cover cross-Region inference.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterable
from dataclasses import dataclass, field
from typing import Any

from strands.models.model import Model

from .spend import NOVA, SpendCap, nova_usd

REGION = "us-east-1"
MODEL_IDS = {
    "micro": "amazon.nova-micro-v1:0",
    "lite": "amazon.nova-lite-v1:0",
    "pro": "amazon.nova-pro-v1:0",
}
BRAIN_MODEL = {"quiet_answer": "micro", "look": "lite", "read_closely": "pro"}
MAX_TOKENS = 160
CHARS_PER_TOKEN = 3.0  # worst case for English; real usage is logged per call
IMAGE_TOKENS = 3000  # worst-case allowance per attached image


@dataclass
class Call:
    model: str
    usd_ceiling: float
    started: float
    first_token_ms: float | None = None
    total_ms: float | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    error: str | None = None

    @property
    def usd(self) -> float:
        return nova_usd(self.model, self.tokens_in, self.tokens_out)


def _text_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, dict):
        return sum(_text_chars(v) for k, v in content.items() if k != "image")
    if isinstance(content, list):
        return sum(_text_chars(v) for v in content)
    return 0


def _images(messages: Any) -> int:
    return sum(
        1
        for m in messages or []
        for block in m.get("content", [])
        if isinstance(block, dict) and "image" in block
    )


def worst_case_usd(model: str, messages: Any, system_prompt: Any, max_tokens: int) -> float:
    tokens_in = (_text_chars(messages) + _text_chars(system_prompt)) / CHARS_PER_TOKEN + 50
    tokens_in += _images(messages) * IMAGE_TOKENS
    return nova_usd(model, int(tokens_in) + 1, max_tokens)


class CappedModel(Model):
    """A Strands model that charges the spend cap before every call."""

    def __init__(self, inner: Model, model: str, cap: SpendCap, max_tokens: int = MAX_TOKENS) -> None:
        if model not in NOVA:
            raise ValueError(f"unknown price key {model!r}")
        self.inner = inner
        self.model = model
        self.cap = cap
        self.max_tokens = max_tokens
        self.calls: list[Call] = []

    @property
    def stateful(self) -> bool:
        return False

    def update_config(self, **model_config: Any) -> None:
        self.inner.update_config(**model_config)

    def get_config(self) -> Any:
        return self.inner.get_config()

    def structured_output(self, output_model: Any, prompt: Any, system_prompt: Any = None, **kwargs: Any):
        usd = worst_case_usd(self.model, prompt, system_prompt, self.max_tokens)
        self.cap.charge(usd, f"bedrock {self.model} structured")
        return self.inner.structured_output(output_model, prompt, system_prompt=system_prompt, **kwargs)

    async def stream(
        self, messages: Any, tool_specs: Any = None, system_prompt: Any = None, **kwargs: Any
    ) -> AsyncIterable[Any]:
        usd = worst_case_usd(self.model, messages, system_prompt, self.max_tokens)
        self.cap.charge(usd, f"bedrock {self.model} images={_images(messages)}")
        call = Call(self.model, round(usd, 6), time.perf_counter())
        self.calls.append(call)
        try:
            async for event in self.inner.stream(
                messages, tool_specs=tool_specs, system_prompt=system_prompt, **kwargs
            ):
                if call.first_token_ms is None and "contentBlockDelta" in event:
                    if "text" in event["contentBlockDelta"].get("delta", {}):
                        call.first_token_ms = (time.perf_counter() - call.started) * 1000
                usage = event.get("metadata", {}).get("usage") if isinstance(event, dict) else None
                if usage:
                    call.tokens_in = int(usage.get("inputTokens", 0))
                    call.tokens_out = int(usage.get("outputTokens", 0))
                yield event
        except Exception as exc:
            call.error = type(exc).__name__
            raise
        finally:
            call.total_ms = (time.perf_counter() - call.started) * 1000


@dataclass
class Brains:
    """One capped model per Nova size; ``calls`` collects every call for the ledger."""

    micro: CappedModel
    lite: CappedModel
    pro: CappedModel
    extra: dict[str, CappedModel] = field(default_factory=dict)

    def all(self) -> list[CappedModel]:
        return [self.micro, self.lite, self.pro, *self.extra.values()]

    def take_calls(self) -> list[Call]:
        out: list[Call] = []
        for m in self.all():
            out.extend(m.calls)
            m.calls = []
        return out


def bedrock_brains(cap: SpendCap, *, region: str = REGION, max_tokens: int = MAX_TOKENS) -> Brains:
    """Real Bedrock models (paid). Each Brains holds its own instances: ModelRouter rejects shared ones."""
    from strands.models.bedrock import BedrockModel

    def make(key: str) -> CappedModel:
        inner = BedrockModel(
            model_id=MODEL_IDS[key], region_name=region, max_tokens=max_tokens, temperature=0.2
        )
        return CappedModel(inner, key, cap, max_tokens)

    return Brains(make("micro"), make("lite"), make("pro"))
