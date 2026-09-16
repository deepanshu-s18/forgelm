"""Tests for shared_schemas: validation gates that training data must pass.

These run fully offline (no GPU, no API). They are the ground truth for
what the data pipeline accepts and what the model must produce.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_schemas.tool_schemas import (
    UNIVERSE_TICKERS,
    HypothesisToolCall,
    OHLCVRequest,
)


class TestOHLCVRequest:
    def test_valid_request(self):
        r = OHLCVRequest(ticker="AAPL", start="2020-01-01", end="2021-01-01")
        assert r.ticker == "AAPL"

    def test_nonuniverse_ticker_rejected(self):
        with pytest.raises(Exception, match="not in universe"):
            OHLCVRequest(ticker="FAKE", start="2020-01-01", end="2021-01-01")

    def test_end_before_start_rejected(self):
        with pytest.raises(Exception):
            OHLCVRequest(ticker="AAPL", start="2022-01-01", end="2020-01-01")

    def test_future_date_rejected(self):
        with pytest.raises(Exception, match="beyond data range"):
            OHLCVRequest(ticker="AAPL", start="2020-01-01", end="2030-01-01")

    def test_lowercase_ticker_rejected(self):
        with pytest.raises(Exception):
            OHLCVRequest(ticker="aapl", start="2020-01-01", end="2021-01-01")


class TestHypothesisToolCall:
    def _valid(self, n_hyp: int = 1) -> dict:
        tool_calls = [{"tool_name": "get_ohlcv",
                       "arguments": {"ticker": "AAPL", "start": "2020-01-01", "end": "2021-01-01"}}]
        if n_hyp > 1:
            tool_calls.append({"tool_name": "apply_bonferroni",
                                "arguments": {"p_values": [0.01, 0.05], "alpha": 0.05}})
        return {
            "user_query": "Test post-earnings drift for AAPL over 10 sessions.",
            "tool_calls": tool_calls,
            "n_hypotheses": n_hyp,
        }

    def test_single_hypothesis_no_correction_ok(self):
        h = HypothesisToolCall(**self._valid(1))
        assert h.n_hypotheses == 1

    def test_multi_hypothesis_requires_correction(self):
        d = self._valid(2)
        # strip the correction tool
        d["tool_calls"] = [tc for tc in d["tool_calls"]
                           if tc["tool_name"] not in {"apply_bonferroni", "apply_bh_fdr"}]
        with pytest.raises(Exception, match="correction tool"):
            HypothesisToolCall(**d)

    def test_multi_hypothesis_with_bh_ok(self):
        d = self._valid(2)
        d["tool_calls"][-1]["tool_name"] = "apply_bh_fdr"
        d["tool_calls"][-1]["arguments"] = {"p_values": [0.01, 0.05], "q": 0.10}
        h = HypothesisToolCall(**d)
        assert h.n_hypotheses == 2

    def test_empty_query_rejected(self):
        d = self._valid()
        d["user_query"] = "Short"
        with pytest.raises(Exception):
            HypothesisToolCall(**d)

    def test_no_tool_calls_rejected(self):
        d = self._valid()
        d["tool_calls"] = []
        with pytest.raises(Exception):
            HypothesisToolCall(**d)


class TestDPOPairGeneration:
    """Offline test for DPO pair injection logic."""

    def test_wrong_ticker_injection(self):
        from forgelm.data_pipeline.gen_dpo_pairs import _inject_wrong_ticker
        rng = random.Random(42)
        sample = {
            "user_query": "Test drift",
            "tool_calls": [{"tool_name": "get_ohlcv",
                             "arguments": {"ticker": "AAPL"}}],
            "n_hypotheses": 1,
        }
        bad = _inject_wrong_ticker(sample, rng)
        assert bad is not None
        assert bad["tool_calls"][0]["arguments"]["ticker"] not in UNIVERSE_TICKERS

    def test_missing_correction_injection(self):
        from forgelm.data_pipeline.gen_dpo_pairs import _inject_missing_correction
        rng = random.Random(42)
        sample = {
            "user_query": "Test drift",
            "tool_calls": [
                {"tool_name": "run_ttest", "arguments": {}},
                {"tool_name": "apply_bonferroni", "arguments": {"p_values": [0.01], "alpha": 0.05}},
            ],
            "n_hypotheses": 3,
        }
        bad = _inject_missing_correction(sample, rng)
        assert bad is not None
        names = {tc["tool_name"] for tc in bad["tool_calls"]}
        assert "apply_bonferroni" not in names and "apply_bh_fdr" not in names

    def test_missing_correction_skips_single_hyp(self):
        from forgelm.data_pipeline.gen_dpo_pairs import _inject_missing_correction
        rng = random.Random(42)
        sample = {"tool_calls": [], "n_hypotheses": 1}
        assert _inject_missing_correction(sample, rng) is None

    def test_hallucinated_date_injection(self):
        from forgelm.data_pipeline.gen_dpo_pairs import _inject_hallucinated_date
        rng = random.Random(42)
        sample = {
            "tool_calls": [{"tool_name": "get_ohlcv",
                            "arguments": {"start": "2020-01-01", "end": "2021-01-01"}}],
            "n_hypotheses": 1,
        }
        bad = _inject_hallucinated_date(sample, rng)
        assert bad is not None
        # should contain a date outside 2015-2025
        for tc in bad["tool_calls"]:
            for k in ("start", "end"):
                v = tc["arguments"].get(k, "")
                if v:
                    year = int(v.split("-")[0])
                    assert year < 2015 or year > 2025, f"date {v} is within valid range"


class TestScoringFunctions:
    """Offline tests for eval metric scoring logic."""

    def test_tool_correct_matches_reference(self):
        from forgelm.eval.run_eval import score_single
        ref = {"tool_calls": [{"tool_name": "get_ohlcv"}], "n_hypotheses": 1}
        pred = {"tool_calls": [{"tool_name": "get_ohlcv"}], "n_hypotheses": 1}
        s = score_single(pred, ref)
        assert s["tool_correct"] == 1

    def test_tool_correct_wrong_tool(self):
        from forgelm.eval.run_eval import score_single
        ref = {"tool_calls": [{"tool_name": "get_ohlcv"}], "n_hypotheses": 1}
        pred = {"tool_calls": [{"tool_name": "run_ttest"}], "n_hypotheses": 1}
        s = score_single(pred, ref)
        assert s["tool_correct"] == 0

    def test_hallucination_detected(self):
        from forgelm.eval.run_eval import score_single
        ref = {"tool_calls": [], "n_hypotheses": 1}
        pred = {"tool_calls": [{"tool_name": "get_ohlcv",
                                 "arguments": {"ticker": "FAKE123"}}],
                "n_hypotheses": 1}
        s = score_single(pred, ref)
        assert s["hallucination"] == 1

    def test_no_hallucination_universe_ticker(self):
        from forgelm.eval.run_eval import score_single
        ref = {"tool_calls": [], "n_hypotheses": 1}
        pred = {"tool_calls": [{"tool_name": "get_ohlcv",
                                 "arguments": {"ticker": "AAPL"}}],
                "n_hypotheses": 1}
        s = score_single(pred, ref)
        assert s["hallucination"] == 0

    def test_correction_required_when_multi_hyp(self):
        from forgelm.eval.run_eval import score_single
        ref = {"tool_calls": [], "n_hypotheses": 3}
        pred = {"tool_calls": [{"tool_name": "run_ttest", "arguments": {}}],
                "n_hypotheses": 3}
        s = score_single(pred, ref)
        assert s["correction_ok"] == 0

    def test_correction_present_scores_1(self):
        from forgelm.eval.run_eval import score_single
        ref = {"tool_calls": [], "n_hypotheses": 3}
        pred = {"tool_calls": [
            {"tool_name": "run_ttest", "arguments": {}},
            {"tool_name": "apply_bh_fdr", "arguments": {"p_values": [0.01], "q": 0.1}},
        ], "n_hypotheses": 3}
        s = score_single(pred, ref)
        assert s["correction_ok"] == 1

    def test_bootstrap_ci_shape(self):
        from forgelm.eval.run_eval import bootstrap_ci
        lo, hi = bootstrap_ci([0, 0, 1, 1, 1, 0, 1, 1], seed=42)
        assert 0.0 <= lo <= hi <= 1.0

    def test_mcnemar_p_no_difference(self):
        from forgelm.eval.run_eval import mcnemar_p
        # both models identical → p should be large
        a = [1, 0, 1, 0, 1, 0]
        p = mcnemar_p(a, a)
        assert p == pytest.approx(1.0) or p >= 0.5

    def test_mcnemar_p_significant_difference(self):
        from forgelm.eval.run_eval import mcnemar_p
        # model A always correct, B always wrong
        a = [1] * 50
        b = [0] * 50
        p = mcnemar_p(a, b)
        assert p < 0.05


# ─── Decontamination tests ────────────────────────────────────────────────────

class TestDecontaminate:
    """8-gram overlap detection removes contaminated eval samples."""

    def _make_jsonl(self, path, records):
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    def test_clean_sample_passes(self, tmp_path):
        from forgelm.data_pipeline.decontaminate import decontaminate
        train = tmp_path / "train.jsonl"
        self._make_jsonl(train, [
            {"user_query": "Do earnings beats in large cap tech stocks cause drift?",
             "n_hypotheses": 1, "tool_calls": [{"tool_name": "get_ohlcv"}]},
        ])
        eval_in = tmp_path / "eval_raw.jsonl"
        self._make_jsonl(eval_in, [
            {"user_query": "Completely different question about reversal momentum signal",
             "n_hypotheses": 1, "tool_calls": [{"tool_name": "get_ohlcv"}]},
        ])
        eval_out = tmp_path / "eval_clean.jsonl"
        stats = decontaminate([train], eval_in, eval_out)
        assert stats["n_clean"] == 1
        assert stats["n_flagged"] == 0

    def test_contaminated_sample_removed(self, tmp_path):
        from forgelm.data_pipeline.decontaminate import decontaminate
        # Plant exact 8-gram from train into eval
        shared_phrase = "earnings beat large cap tech momentum reversal signal alpha"
        train = tmp_path / "train.jsonl"
        self._make_jsonl(train, [{"user_query": shared_phrase}])
        eval_in = tmp_path / "eval_raw.jsonl"
        # eval sample contains the exact same 8-gram
        self._make_jsonl(eval_in, [{"user_query": shared_phrase + " additional context"}])
        eval_out = tmp_path / "eval_clean.jsonl"
        stats = decontaminate([train], eval_in, eval_out)
        assert stats["n_flagged"] == 1
        assert stats["n_clean"] == 0

    def test_short_sample_no_false_positive(self, tmp_path):
        """Samples with < 8 words can't have 8-grams; should never be flagged."""
        from forgelm.data_pipeline.decontaminate import decontaminate
        train = tmp_path / "train.jsonl"
        self._make_jsonl(train, [{"user_query": "short"}])
        eval_in = tmp_path / "eval_raw.jsonl"
        self._make_jsonl(eval_in, [{"user_query": "also short"}])
        eval_out = tmp_path / "eval_clean.jsonl"
        stats = decontaminate([train], eval_in, eval_out)
        assert stats["n_clean"] == 1
