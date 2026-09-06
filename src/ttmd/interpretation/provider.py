"""LLM provider interface + a no-API stub, so the system runs and is testable
without credentials. Swap in a real adapter (Bedrock/OpenAI/Anthropic) later.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        """Return the model's text completion for a system + user prompt.

        max_tokens: optional per-call output budget. Large structured outputs
        (e.g. describing many fields as JSON) need a higher budget than the
        default to avoid truncation.
        """
        ...


class StubProvider(LLMProvider):
    """Deterministic, offline stand-in. Produces a plainly-labeled, generic
    hypothesis so the pipeline works end-to-end with no API. NOT real reasoning —
    replace with a real provider for genuine root-cause suggestions.
    """

    def complete(self, system: str, user: str, max_tokens: int | None = None) -> str:
        # No real reasoning offline. Echo a clearly-labeled placeholder so the
        # report STRUCTURE (per-relationship + cross-relationship) is visible.
        # Connect a real provider (TTMD_LLM=bedrock) for genuine hypotheses.
        if user.strip().startswith("These signals are all related"):
            return ("(stub) Multiple signals share a common driver — a real LLM "
                    "would propose a single physical mechanism here.")
        return ("(stub) Connect an LLM provider for an asset-specific hypothesis.")


class BedrockProvider(LLMProvider):  # pragma: no cover - needs AWS creds
    """Amazon Bedrock adapter using the Converse API.

    Notes learned from the live account:
    - eu-west-1 requires the regional inference-profile prefix, e.g.
      'eu.anthropic.claude-haiku-4-5-20251001-v1:0' (bare model IDs 404 on Converse).
    - Configurable via env: TTMD_BEDROCK_REGION, TTMD_BEDROCK_MODEL.
    """

    DEFAULT_REGION = "eu-west-1"
    DEFAULT_MODEL = "eu.anthropic.claude-haiku-4-5-20251001-v1:0"

    def __init__(self, model_id: str | None = None, region: str | None = None,
                 max_tokens: int = 1024):
        import boto3  # lazy: stub path needs no boto3
        self._region = region or os.environ.get("TTMD_BEDROCK_REGION", self.DEFAULT_REGION)
        self._model_id = model_id or os.environ.get("TTMD_BEDROCK_MODEL", self.DEFAULT_MODEL)
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
        return resp["output"]["message"]["content"][0]["text"]


def get_provider() -> LLMProvider:
    """Select a provider from env (TTMD_LLM=bedrock|stub). Defaults to stub."""
    kind = os.environ.get("TTMD_LLM", "stub").lower()
    if kind == "bedrock":
        return BedrockProvider()
    return StubProvider()
