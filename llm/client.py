"""
Unified LLM client.

Supports:
- Anthropic (Claude Opus / Sonnet / Haiku) via ANTHROPIC_API_KEY
- Dry-run mode: returns canned responses, useful for plumbing tests without API

Usage:
    llm = LLMClient.from_env(model="claude-sonnet-4-5")
    resp = llm.complete(system="...", messages=[{"role":"user","content":"..."}])
    # resp is a Response with .text, .usage, .raw

JSON helper:
    obj = llm.complete_json(system=..., messages=..., schema=MyPydanticModel)
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)


T = TypeVar("T", bound=BaseModel)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


@dataclass
class Response:
    text: str
    usage: Usage
    raw: Any = None
    model: str = ""


class LLMClient:
    """Abstract LLM client."""

    def __init__(self, model: str, dry_run: bool = False):
        self.model = model
        self.dry_run = dry_run
        self.total_usage = Usage()

    # ---- basic completion ----
    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> Response:
        raise NotImplementedError

    # ---- JSON-typed completion ----
    def complete_json(
        self,
        system: str,
        messages: list[dict],
        schema: Type[T],
        max_tokens: int = 2048,
        temperature: float = 0.2,
        max_retries: int = 2,
    ) -> T:
        """
        Ask the LLM for JSON matching `schema` (a Pydantic model).

        Appends an instruction to the system prompt asking for JSON only,
        parses the response, retries on ValidationError with the error message.
        """
        schema_json = json.dumps(
            schema.model_json_schema(), indent=2
        )
        augmented_system = (
            system.rstrip()
            + "\n\nReturn ONLY a JSON object matching this schema:\n"
            + schema_json
            + "\nDo not include any text before or after the JSON. Do not wrap in markdown code fences."
        )

        last_error = None
        for attempt in range(max_retries + 1):
            resp = self.complete(
                system=augmented_system,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = resp.text.strip()
            # Strip common code-fence wrappers
            m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if m:
                text = m.group(1).strip()
            try:
                obj = json.loads(text)
                return schema.model_validate(obj)
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                log.warning("JSON parse/validation failed (attempt %d): %s", attempt, e)
                if attempt < max_retries:
                    messages = messages + [
                        {"role": "assistant", "content": text},
                        {"role": "user",
                         "content": f"That was not valid JSON matching the schema. Error: {e}\nTry again. Output ONLY the JSON object."}
                    ]
                    continue
                raise RuntimeError(f"LLM failed to produce valid JSON after {max_retries+1} attempts: {last_error}\nLast response:\n{text[:500]}")

    # ---- factory ----
    @classmethod
    def from_env(
        cls,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        dry_run: bool = False,
    ) -> "LLMClient":
        """
        Pick the right backend based on environment.

        Precedence:
          1. If dry_run=True or LIMIT_TREE_DRY_RUN=1, return DryRunClient.
          2. If provider is explicitly set, honor it.
          3. If LIMIT_TREE_PROVIDER=bedrock or AWS credentials look usable,
             return BedrockSyncClient.
          4. If ANTHROPIC_API_KEY is set, return AnthropicClient.
          5. Else, fall back to DryRunClient with a warning.
        """
        if dry_run or os.environ.get("LIMIT_TREE_DRY_RUN") == "1":
            return DryRunClient(model=model or "dry-run")

        provider = provider or os.environ.get("LIMIT_TREE_PROVIDER")

        if provider == "bedrock" or (
            provider is None
            and (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION"))
            and not os.environ.get("ANTHROPIC_API_KEY")
        ):
            try:
                from llm.bedrock import BedrockSyncClient
                return BedrockSyncClient(
                    model=model or os.environ.get(
                        "LIMIT_TREE_MODEL",
                        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
                    ),
                )
            except Exception as e:
                log.warning("Bedrock client unavailable: %s", e)

        if provider == "anthropic" or os.environ.get("ANTHROPIC_API_KEY"):
            return AnthropicClient(
                model=model or os.environ.get("LIMIT_TREE_MODEL", "claude-sonnet-4-5-20250929"),
            )

        log.warning("No LLM credentials found; falling back to DryRunClient")
        return DryRunClient(model=model or "dry-run")


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

class AnthropicClient(LLMClient):
    def __init__(self, model: str = "claude-sonnet-4-5-20250929"):
        super().__init__(model=model, dry_run=False)
        from anthropic import Anthropic
        self._client = Anthropic()

    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> Response:
        backoff = 2.0
        for attempt in range(4):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    system=system,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = "".join(
                    b.text for b in resp.content if getattr(b, "type", None) == "text"
                )
                usage = Usage(
                    input_tokens=resp.usage.input_tokens,
                    output_tokens=resp.usage.output_tokens,
                )
                self.total_usage = self.total_usage.add(usage)
                return Response(text=text, usage=usage, raw=resp, model=self.model)
            except Exception as e:
                log.warning("Anthropic call failed (attempt %d): %s", attempt, e)
                if attempt < 3:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise


# ---------------------------------------------------------------------------
# Dry-run client (for testing without API)
# ---------------------------------------------------------------------------

class DryRunClient(LLMClient):
    """
    Returns deterministic canned responses. Useful for plumbing tests and CI.
    The response is a JSON object that is *schema-agnostic* and will only
    succeed with schemas that have all optional fields (or use placeholders
    via `set_canned`).
    """

    _canned: dict = {}   # by prompt hash

    def __init__(self, model: str = "dry-run"):
        super().__init__(model=model, dry_run=True)

    @classmethod
    def set_canned(cls, canned: dict):
        cls._canned = canned

    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> Response:
        last_user = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user = m["content"]
                break
        key = hash(last_user)
        if key in self._canned:
            text = self._canned[key]
        else:
            # Fallback: return an empty JSON object (will fail schema validation
            # for non-trivial schemas, but that's fine — this client is for
            # plumbing sanity, not correctness.)
            text = "{}"
        return Response(text=text, usage=Usage(1, 1), model=self.model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    llm = LLMClient.from_env()
    print("Using:", type(llm).__name__, "model=", llm.model)
    if not isinstance(llm, DryRunClient):
        resp = llm.complete(
            system="You are a helpful assistant. Be terse.",
            messages=[{"role": "user", "content": "Say hi in 5 words."}],
            max_tokens=50,
        )
        print("Response:", resp.text)
        print("Usage:", resp.usage)
