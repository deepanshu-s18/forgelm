"""ForgeLM → AlphaForge integration adapter.

Exposes ForgeLM as an OpenAI-compatible endpoint that AlphaForge's
HypothesisAgent can call when `llm_backend='forgelm'` is set.

Replaces the Claude backend for offline/air-gapped environments.
Responses go through the SAME HypothesisToolCall schema validation as
the Claude backend — no special-casing, same honesty guarantees.

Usage (requires vllm running on :8000):
    python -m forgelm.serving.alphaforge_adapter \
        --model checkpoints/stage3_dpo/final \
        --port 8001
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from shared_schemas.tool_schemas import HypothesisToolCall

log = structlog.get_logger("forgelm.adapter")

SYSTEM_PROMPT = (
    "You are a financial research agent with access to market data and "
    "statistical tools. When asked to test hypotheses, return a valid JSON "
    "tool-call sequence. When testing >1 hypothesis, always include "
    "apply_bonferroni or apply_bh_fdr."
)


class ForgeLMClient:
    """Drop-in replacement for alphaforge.tools.llm.ClaudeClient.

    Implements the same interface so AlphaForge's orchestrator can
    switch backends without touching agent logic.
    """

    def __init__(self, base_url: str = "http://localhost:8000/v1",
                 model: str = "forgelm", temperature: float = 0.0):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai for ForgeLM serving") from e
        self._client = OpenAI(base_url=base_url, api_key="none")
        self._model = model
        self._temperature = temperature
        self.rejected_outputs = 0
        self.notes: list[str] = []

    def refine_hypotheses(self, seed_query: str, hypotheses: list) -> list:
        """Reword hypotheses via structured JSON; same contract as ClaudeClient."""
        from alphaforge.state.schema import Hypothesis

        payload = json.dumps({
            "seed_query": seed_query,
            "hypotheses": [{"id": h.id, "statement": h.statement, "family": h.family}
                           for h in hypotheses],
            "n_hypotheses": len(hypotheses),
        })
        resp = self._client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": payload},
            ],
        )
        raw = resp.choices[0].message.content or ""
        try:
            # Validate against shared schema before trusting
            data = json.loads(raw)
            HypothesisToolCall(**{**data, "user_query": seed_query, "n_hypotheses": len(hypotheses)})  # noqa: E501
        except Exception as e:
            self.rejected_outputs += 1
            self.notes.append(f"forgelm refinement rejected: {e}")
            log.warning("forgelm_refinement_rejected", error=str(e)[:120])
            return hypotheses  # fall back to templates

        by_id = {h.id: h for h in hypotheses}
        out = list(hypotheses)
        for item in data.get("hypotheses", []):
            hid = item.get("id")
            if hid in by_id:
                try:
                    base = by_id[hid]
                    candidate = base.model_copy(update={"statement": str(item.get("statement", ""))})  # noqa: E501
                    Hypothesis.model_validate(candidate.model_dump())
                    out[out.index(base)] = candidate
                except Exception:
                    self.rejected_outputs += 1
        return out

    def polish_discussion(self, context: str) -> str | None:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                temperature=0.3,
                max_tokens=600,
                messages=[
                    {"role": "system", "content": "Polish this financial research discussion. "
                     "Be concise and grounded. Do not add claims not in the context."},
                    {"role": "user", "content": context},
                ],
            )
            return resp.choices[0].message.content.strip() or None
        except Exception as e:
            self.notes.append(f"polish failed: {e}")
            return None


def main():
    """Start vllm OpenAI-compatible server for ForgeLM."""
    import argparse
    import subprocess
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="checkpoints/stage3_dpo/final")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--max-model-len", type=int, default=2048)
    args = p.parse_args()
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(args.port),
        "--max-model-len", str(args.max_model_len),
        "--dtype", "bfloat16",
    ]
    log.info("starting_vllm", command=" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
