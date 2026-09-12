"""LLM provider interface + a no-API stub, so the system runs and is testable
without credentials. Swap in a real adapter (Bedrock/OpenAI/Anthropic) later.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod


def _env(name: str, default: str | None = None) -> str | None:
    """Read a GALENE_<name> env var (e.g. GALENE_LLM=bedrock)."""
    return os.environ.get(f"GALENE_{name}", default)


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        """Return the model's text completion for a system + user prompt.

        max_tokens: optional per-call output budget. Large structured outputs
        (e.g. describing many fields as JSON) need a higher budget than the
        default to avoid truncation.
        """
        ...

    @property
    def synthesizes(self) -> bool:
        """Whether this provider does REAL free-form synthesis/reasoning.

        A real LLM returns True. The offline StubProvider returns False: it
        emits a fixed placeholder, so callers that HAVE a deterministic fallback
        (summarize, reason) should prefer that fallback rather than surface the
        placeholder. (A truthiness check on the reply can't tell them apart —
        the stub's placeholder is non-empty text.)
        """
        return True


class StubProvider(LLMProvider):
    """Deterministic, offline stand-in. Produces a plainly-labeled, generic
    hypothesis so the pipeline works end-to-end with no API. NOT real reasoning —
    replace with a real provider for genuine root-cause suggestions.
    """

    synthesizes = False   # offline placeholder, not real synthesis (see base)

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        # No real reasoning offline. Echo a clearly-labeled placeholder so the
        # report STRUCTURE (per-relationship + cross-relationship) is visible.
        # Connect a real provider (GALENE_LLM=bedrock) for genuine hypotheses.
        if user.strip().startswith("These signals are all related"):
            return ("(stub) Multiple signals share a common driver — a real LLM "
                    "would propose a single physical mechanism here.")
        return ("(stub) Connect an LLM provider for an asset-specific hypothesis.")


class BedrockProvider(LLMProvider):  # pragma: no cover - needs AWS creds
    """Amazon Bedrock adapter using the Converse API.

    Notes learned from the live account:
    - eu-west-1 requires the regional inference-profile prefix, e.g.
      'eu.anthropic.claude-haiku-4-5-20251001-v1:0' (bare model IDs 404 on Converse).
    - Configurable via env: GALENE_BEDROCK_REGION, GALENE_BEDROCK_MODEL.
    """

    DEFAULT_REGION = "eu-west-1"
    DEFAULT_MODEL = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"

    def __init__(self, model_id: str | None = None, region: str | None = None,
                 max_tokens: int = 1024):
        import boto3  # lazy: stub path needs no boto3
        self._region = region or _env("BEDROCK_REGION", self.DEFAULT_REGION)
        self._model_id = model_id or _env("BEDROCK_MODEL", self.DEFAULT_MODEL)
        self._client = boto3.client("bedrock-runtime", region_name=self._region)
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        resp = self._client.converse(
            modelId=self._model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens or self._max_tokens,
                             "temperature": 0.2},
        )
        # expose real token usage so a UsageMeter wrapper can tally exact counts
        self._last_usage = resp.get("usage")
        return resp["output"]["message"]["content"][0]["text"]


# Approximate Amazon Bedrock on-demand price for Claude Haiku 4.5 (per 1K tokens,
# USD). ROUGH — pricing changes and varies by region; used only for a cost
# ESTIMATE the operator can sanity-check, never billed. Override via env.
_HAIKU_INPUT_PER_1K = float(_env("PRICE_IN_PER_1K", "0.001"))
_HAIKU_OUTPUT_PER_1K = float(_env("PRICE_OUT_PER_1K", "0.005"))
# fallback token estimate when the backend doesn't report usage (~4 chars/token)
_CHARS_PER_TOKEN = 4


class UsageMeter(LLMProvider):
    """Wraps any provider and tallies calls + input/output tokens + a rough cost
    estimate, WITHOUT changing behavior. Bedrock's Converse response reports real
    token usage; for the stub (or if usage is absent) we estimate from text length
    (~4 chars/token). Read totals via .summary(). Purely additive observability."""

    def __init__(self, inner: LLMProvider):
        self.inner = inner
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.last_usage = None

    @property
    def synthesizes(self) -> bool:
        # delegate to the wrapped provider so metering doesn't mask a stub
        return getattr(self.inner, "synthesizes", True)

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        self.calls += 1
        # let a Bedrock inner report exact usage via a side channel if present
        usage_before = getattr(self.inner, "_last_usage", None)
        out = self.inner.complete(system, user, max_tokens=max_tokens)
        usage = getattr(self.inner, "_last_usage", None)
        if usage and usage is not usage_before:
            self.input_tokens += int(usage.get("inputTokens", 0))
            self.output_tokens += int(usage.get("outputTokens", 0))
            self.last_usage = usage
        else:
            # estimate from character length (input = system+user, output = out)
            self.input_tokens += max(1, (len(system) + len(user)) // _CHARS_PER_TOKEN)
            self.output_tokens += max(1, len(out) // _CHARS_PER_TOKEN)
        return out

    def estimated_cost_usd(self) -> float:
        return round(self.input_tokens / 1000 * _HAIKU_INPUT_PER_1K
                     + self.output_tokens / 1000 * _HAIKU_OUTPUT_PER_1K, 4)

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd(),
            "tokens_estimated": self.last_usage is None,
        }

    def summary_line(self) -> str:
        s = self.summary()
        approx = "~" if s["tokens_estimated"] else ""
        return (f"LLM usage: {s['calls']} call(s), {approx}{s['total_tokens']:,} tokens "
                f"({s['input_tokens']:,} in / {s['output_tokens']:,} out), "
                f"est. cost {approx}${s['estimated_cost_usd']:.4f}"
                + (" [token counts estimated from text length]"
                   if s["tokens_estimated"] else ""))


def get_provider(meter: bool = False) -> LLMProvider:
    """Select a provider from env (GALENE_LLM=bedrock|stub). Defaults to stub.
    meter=True wraps it in a UsageMeter for token/cost logging."""
    kind = (_env("LLM", "stub") or "stub").lower()
    inner = BedrockProvider() if kind == "bedrock" else StubProvider()
    return UsageMeter(inner) if meter else inner
