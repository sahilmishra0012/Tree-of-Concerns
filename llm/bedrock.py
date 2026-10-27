"""
AWS Bedrock backends for LimitTree.

Two clients:

  1. BedrockSyncClient - synchronous Bedrock calls via invoke_model.
     Use for interactive development, debug, small runs.

  2. BedrockBatchClient - batches requests. Call .submit(req) to enqueue;
     call .run() to flush all pending requests via Bedrock batch inference
     (S3 input -> batch job -> S3 output). ~50% cheaper than on-demand.

Both implement the same `LLMClient` interface (complete / complete_json).

Bedrock batch inference:
  - Input: JSONL in S3, one record per line with {"recordId", "modelInput"}
  - Output: JSONL in S3 with {"recordId", "modelOutput"}
  - Minimum 100 records per job in most regions.
  - Completion: minutes to hours depending on load and size.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from llm.client import LLMClient, Response, Usage

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bedrock Claude "invoke_model" request/response helpers
# ---------------------------------------------------------------------------

def build_invoke_body(
    system: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> dict:
    """Payload shape for Anthropic Claude on Bedrock invoke_model."""
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
        "temperature": temperature,
    }


def parse_invoke_output(output: dict) -> tuple[str, Usage]:
    """Extract text + usage from an Anthropic-on-Bedrock response."""
    # content is list of blocks like {"type": "text", "text": "..."}
    text = ""
    for block in output.get("content", []) or []:
        if block.get("type") == "text":
            text += block.get("text", "")
    usage_raw = output.get("usage") or {}
    usage = Usage(
        input_tokens=usage_raw.get("input_tokens", 0),
        output_tokens=usage_raw.get("output_tokens", 0),
    )
    return text, usage


# ---------------------------------------------------------------------------
# Synchronous Bedrock client
# ---------------------------------------------------------------------------

class BedrockSyncClient(LLMClient):
    """
    Sync Bedrock client using invoke_model. One API call per complete().
    """

    def __init__(
        self,
        model: str,
        region: Optional[str] = None,
        profile: Optional[str] = None,
    ):
        super().__init__(model=model, dry_run=False)
        import boto3
        session = boto3.Session(profile_name=profile) if profile else boto3.Session()
        self.region = region or session.region_name or "us-east-1"
        self._client = session.client("bedrock-runtime", region_name=self.region)

    def complete(
        self,
        system: str,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> Response:
        body = build_invoke_body(system, messages, max_tokens, temperature)
        backoff = 2.0
        last_err = None
        for attempt in range(4):
            try:
                resp = self._client.invoke_model(
                    modelId=self.model,
                    body=json.dumps(body).encode("utf-8"),
                    accept="application/json",
                    contentType="application/json",
                )
                output = json.loads(resp["body"].read())
                text, usage = parse_invoke_output(output)
                self.total_usage = self.total_usage.add(usage)
                return Response(text=text, usage=usage, raw=output, model=self.model)
            except Exception as e:
                last_err = e
                log.warning("Bedrock invoke failed (attempt %d): %s", attempt, e)
                if attempt < 3:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                raise



if __name__ == "__main__":
    import sys, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Say hello in one short sentence.")
    ap.add_argument("--model", default="global.anthropic.claude-opus-4-6-v1")
    args = ap.parse_args()
    client = BedrockSyncClient(model_id=args.model)
    try:
        resp = client.complete(system="You are helpful.",
                              messages=[{"role":"user","content":args.prompt}],
                              max_tokens=100, temperature=0.0)
        print(f"Response: {resp.text}")
    except Exception as e:
        print(f"Bedrock sync call failed: {e}", file=sys.stderr)
        sys.exit(1)
