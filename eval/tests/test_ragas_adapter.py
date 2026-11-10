"""RAGAS 适配层的纯函数测试。

刻意只测不联网的部分：`compute_ragas` 要 judge，CI 不能碰；但它外面那层
"重复跑 → 压成 mean/spread → 落侧车"的逻辑是确定性的，必须被测试守住，
否则哪天有人在 `summarize_runs` 里把 spread 算成标准差，没人会发现。
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from runner import ragas_adapter as R
from runner.metrics import QueryOutcome


def outcome(qid: str, answer: str, docs: list[str], *, refused: bool = False) -> QueryOutcome:
    return QueryOutcome(
        query_id=qid,
        category="current_rule",
        as_of="2026-11-08",
        expected_doc_ids=docs,
        expect_refusal=False,
        expect_stale_recall=False,
        answer=answer,
        refused=refused,
        retrieved_doc_ids=docs,
    )


class TestToRagasSamples:
    def test_maps_query_answer_and_contexts(self) -> None:
        rows = [{"query_id": "q01", "query": "转专业条件"}]
        outs = [outcome("q01", "绩点不低于2.5", ["jwb-jxtz-2026-29459"])]
        samples = R.to_ragas_samples(rows, outs)
        assert samples == [
            {
                "user_input": "转专业条件",
                "response": "绩点不低于2.5",
                "retrieved_contexts": ["jwb-jxtz-2026-29459"],
            }
        ]

    def test_skips_rows_without_answer(self) -> None:
        """拒答题没有答案，喂给 RAGAS 只会让 faithfulness 变成 NaN。"""
        rows = [{"query_id": "q01", "query": "a"}, {"query_id": "q06", "query": "b"}]
        outs = [outcome("q01", "有答案", ["d1"]), outcome("q06", "", [], refused=True)]
        assert len(R.to_ragas_samples(rows, outs)) == 1

    def test_falls_back_when_no_recall(self) -> None:
        rows = [{"query_id": "q01", "query": "a"}]
        outs = [outcome("q01", "有答案", [])]
        samples = R.to_ragas_samples(rows, outs)
        assert samples[0]["retrieved_contexts"] == ["（无召回）"]


class TestSummarizeRuns:
    def test_mean_min_max_spread(self) -> None:
        runs = [
            {"faithfulness": 0.80, "answer_relevancy": 0.90},
            {"faithfulness": 0.86, "answer_relevancy": 0.92},
            {"faithfulness": 0.83, "answer_relevancy": 0.91},
        ]
        stats = R.summarize_runs(runs)
        assert stats["faithfulness"] == {
            "mean": 0.83,
            "min": 0.8,
            "max": 0.86,
            "spread": 0.06,
        }
        assert stats["answer_relevancy"]["spread"] == 0.02

    def test_ignores_none_and_keeps_key(self) -> None:
        runs = [{"faithfulness": None}, {"faithfulness": 0.5}]
        stats = R.summarize_runs(runs)
        assert stats["faithfulness"]["mean"] == 0.5
        assert stats["faithfulness"]["spread"] == 0.0

    def test_all_none_returns_empty_stats(self) -> None:
        stats = R.summarize_runs([{"faithfulness": None}, {"faithfulness": None}])
        assert stats["faithfulness"] == {"mean": None, "min": None, "max": None, "spread": None}


class TestVerdict:
    def test_stable_run_is_acceptable(self) -> None:
        stats = {"faithfulness": {"mean": 0.85, "min": 0.84, "max": 0.86, "spread": 0.02}}
        assert "可接受" in R._verdict_text(stats)

    def test_spread_above_threshold_is_rejected(self) -> None:
        """波动 > 0.05 的结果不得用于验收 —— 这条是 Issue #5 的硬要求。"""
        stats = {"faithfulness": {"mean": 0.85, "min": 0.78, "max": 0.9, "spread": 0.12}}
        text = R._verdict_text(stats)
        assert "不得用于验收" in text and "0.1200" in text

    def test_no_scores_is_rejected(self) -> None:
        stats = {"faithfulness": {"mean": None, "min": None, "max": None, "spread": None}}
        assert "不得引用" in R._verdict_text(stats)


class TestBuildRunRecord:
    def test_record_shape(self) -> None:
        stats = {"faithfulness": {"mean": 0.85, "min": 0.84, "max": 0.86, "spread": 0.02}}
        record = R.build_run_record(
            report_name="run-1.json",
            created_at="2026-11-08T12:00:00+08:00",
            llm_model="deepseek-chat",
            embedding_model="text-embedding-v3",
            sample_count=40,
            repeat=3,
            duration_ms=1234,
            runs=[{"faithfulness": 0.85}],
            summary=stats,
        )
        assert record["judge"] == {"llm": "deepseek-chat", "embedding": "text-embedding-v3"}
        assert record["repeat"] == 3
        assert record["summary"] == stats
        assert "verdict" in record


class TestMean:
    def test_nan_becomes_none(self) -> None:
        class Series:
            def mean(self) -> float:
                return float("nan")

        assert R._mean(Series()) is None

    def test_none_series(self) -> None:
        assert R._mean(None) is None

    def test_normal(self) -> None:
        class Series:
            def mean(self) -> float:
                return 0.81234

        assert R._mean(Series()) == 0.8123


class TestUnavailable:
    def test_missing_ragas_raises(self) -> None:
        try:
            import ragas  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("本机装了 ragas，跳过‘未安装’分支")
        with pytest.raises(R.RagasUnavailable):
            R.compute_ragas([{"user_input": "q", "response": "a", "retrieved_contexts": ["c"]}])


def test_math_sanity() -> None:
    """spread 必须是非负数；这里用 math 兜一下，防止未来改成方差没开根。"""
    stats = R.summarize_runs([{"m": 0.1}, {"m": 0.4}])
    value: Any = stats["m"]["spread"]
    assert math.isfinite(value) and value >= 0
