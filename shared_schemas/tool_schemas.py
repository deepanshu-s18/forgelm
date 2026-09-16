"""Shared Pydantic tool schemas — imported by both AlphaForge and ForgeLM.

These schemas define the exact I/O contract for AlphaForge's MCP tools.
ForgeLM's data pipeline validates every generated training sample against
these schemas so 100% of SFT data is machine-validated — no hallucinated
tickers, no out-of-range dates, no missing correction calls.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")

# ── Universe (kept in sync with alphaforge/config/universe.yaml) ────────────
UNIVERSE_TICKERS: set[str] = {
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "INTC",
    "CSCO", "JPM", "BAC", "GS", "MS", "WMT", "XOM", "CVX", "JNJ", "PFE",
    "UNH", "PG", "KO", "PEP", "DIS", "NFLX",
    "XLK", "XLF", "XLY", "XLP", "XLE", "XLV", "XLI", "XLB", "XLRE", "XLU", "XLC",
}
DATA_START = date(2015, 1, 1)
DATA_END = date(2025, 12, 31)


class ToolCallRequest(BaseModel):
    """A single tool invocation produced by the LLM."""
    tool_name: str
    arguments: dict[str, Any]


class OHLCVRequest(BaseModel):
    ticker: str
    start: date
    end: date
    interval: Literal["1d", "1h"] = "1d"

    @field_validator("ticker")
    @classmethod
    def _ticker_valid(cls, v: str) -> str:
        if not _TICKER_RE.match(v):
            raise ValueError(f"invalid ticker: {v!r}")
        if v not in UNIVERSE_TICKERS:
            raise ValueError(f"ticker {v!r} not in universe")
        return v

    @field_validator("end")
    @classmethod
    def _range_ok(cls, v: date, info) -> date:
        start = info.data.get("start")
        if start and v <= start:
            raise ValueError("end must be after start")
        if v > DATA_END:
            raise ValueError(f"end {v} beyond data range {DATA_END}")
        return v


class EarningsRequest(BaseModel):
    ticker: str
    start: date
    end: date

    @field_validator("ticker")
    @classmethod
    def _ticker_valid(cls, v: str) -> str:
        if v not in UNIVERSE_TICKERS:
            raise ValueError(f"ticker {v!r} not in universe")
        return v


class RunTTestRequest(BaseModel):
    sample_a: list[float] = Field(..., min_length=2)
    sample_b: list[float] = Field(..., min_length=2)
    alternative: Literal["two-sided", "less", "greater"] = "two-sided"


class ApplyBonferroniRequest(BaseModel):
    p_values: list[float] = Field(..., min_length=1)
    alpha: float = Field(0.05, gt=0.0, lt=1.0)


class ApplyBHFDRRequest(BaseModel):
    p_values: list[float] = Field(..., min_length=1)
    q: float = Field(0.10, gt=0.0, lt=1.0)


class HypothesisToolCall(BaseModel):
    """A full scenario: query + the correct tool-call sequence."""
    user_query: str = Field(..., min_length=10)
    # assistant must emit at least one statistical test AND correction when n>1
    tool_calls: list[ToolCallRequest] = Field(..., min_length=1)
    n_hypotheses: int = Field(..., ge=1)

    @model_validator(mode="after")
    def _correction_required(self) -> "HypothesisToolCall":
        """When n_hypotheses > 1 the response MUST include a correction tool.

        Using model_validator(mode='after') so both tool_calls and n_hypotheses
        are already populated (field_validator on tool_calls fires before
        n_hypotheses is set, making info.data unreliable for cross-field checks).
        """
        if self.n_hypotheses > 1:
            correction_tools = {"apply_bonferroni", "apply_bh_fdr"}
            used = {tc.tool_name for tc in self.tool_calls}
            if not used & correction_tools:
                raise ValueError(
                    f"n_hypotheses={self.n_hypotheses} > 1 requires a correction tool "
                    f"(apply_bonferroni or apply_bh_fdr); got tools: {used}"
                )
        return self



class EventRecord(BaseModel):
    """Structured event extracted from an earnings-call snippet."""
    ticker: str
    event_type: Literal["earnings_beat", "earnings_miss", "guidance_raise",
                        "guidance_lower", "revenue_beat", "revenue_miss"]
    surprise_pct: float | None = None
    guidance: Literal["raised", "lowered", "maintained", "not_given"] = "not_given"
    sentiment: Literal["positive", "neutral", "negative"] = "neutral"

    @field_validator("ticker")
    @classmethod
    def _ticker_valid(cls, v: str) -> str:
        if v not in UNIVERSE_TICKERS:
            raise ValueError(f"ticker {v!r} not in universe")
        return v
