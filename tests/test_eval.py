"""Tests for ForgeLM evaluation and statistical reporting logic.

Runs completely offline (no GPU, no model download required).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from forgelm.eval.run_eval import bootstrap_ci, mcnemar_p, score_single
from forgelm.eval.stats_report import bootstrap_diff_ci, generate_report


class TestEvalScoring:
    def test_score_single_valid(self):
        ref = {
            "n_hypotheses": 1,
            "tool_calls": [{"tool_name": "get_ohlcv", "arguments": {"ticker": "AAPL"}}],
        }
        pred = {
            "thought": "fetching AAPL data",
            "tool_calls": [
                {
                    "tool_name": "get_ohlcv",
                    "arguments": {"ticker": "AAPL", "start": "2020-01-01", "end": "2021-01-01"},
                }
            ],
        }
        scores = score_single(pred, ref)
        assert scores["tool_correct"] == 1
        assert scores["schema_valid"] == 1
        assert scores["correction_ok"] == 1
        assert scores["hallucination"] == 0

    def test_score_single_hallucination(self):
        ref = {
            "n_hypotheses": 1,
            "tool_calls": [{"tool_name": "get_ohlcv", "arguments": {"ticker": "AAPL"}}],
        }
        pred = {
            "thought": "invalid ticker",
            "tool_calls": [
                {
                    "tool_name": "get_ohlcv",
                    "arguments": {
                        "ticker": "FAKE_TICKER",
                        "start": "2020-01-01",
                        "end": "2021-01-01",
                    },
                }
            ],
        }
        scores = score_single(pred, ref)
        assert scores["hallucination"] == 1

    def test_score_single_multi_hyp_missing_correction(self):
        ref = {
            "n_hypotheses": 3,
            "tool_calls": [{"tool_name": "run_ttest", "arguments": {}}],
        }
        pred = {
            "thought": "missing stats correction",
            "tool_calls": [
                {
                    "tool_name": "run_ttest",
                    "arguments": {"series_a": [1.0, 2.0], "series_b": [2.0, 3.0]},
                }
            ],
        }
        scores = score_single(pred, ref)
        assert scores["correction_ok"] == 0


class TestStatsUtilities:
    def test_bootstrap_ci_coverage(self):
        vals = [1.0] * 50 + [0.0] * 50
        lo, hi = bootstrap_ci(vals, n_boot=500, seed=42)
        assert 0.35 <= lo <= 0.50
        assert 0.50 <= hi <= 0.65

    def test_mcnemar_identical(self):
        a = [1, 0, 1, 1, 0]
        b = [1, 0, 1, 1, 0]
        assert mcnemar_p(a, b) == 1.0

    def test_bootstrap_diff_ci(self):
        a = [1.0] * 100
        b = [0.0] * 100
        lo, hi = bootstrap_diff_ci(a, b, n_boot=500, seed=42)
        assert lo > 0.9
        assert hi <= 1.0


class TestReportGeneration:
    def test_generate_report_from_summary(self):
        results = {
            "stage1": {"loss": 1.74, "steps": 41},
            "stage2": {"loss": 1.69, "steps": 22},
            "stage3": {"loss": 2.70, "steps": 47},
            "eval": {
                "base": {"json_schema_rate": 0.0, "tool_name_recall": 0.0, "ticker_recall": 0.312},
                "sft": {"json_schema_rate": 0.0, "tool_name_recall": 0.0, "ticker_recall": 0.375},
                "dpo": {"json_schema_rate": 0.0, "tool_name_recall": 0.0, "ticker_recall": 0.312},
            },
        }
        with tempfile.TemporaryDirectory() as td:
            out_file = Path(td) / "results.md"
            md = generate_report(results, out_file)
            assert out_file.exists()
            assert "Stage 1: DAPT" in md
            assert "0.3750" in md
