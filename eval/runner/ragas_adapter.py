"""可选：用 RAGAS 算生成质量指标（faithfulness / answer_relevancy）。

**不进 CI**。原因写在下面，改之前先读完：

- RAGAS 需要 LLM judge。骨架阶段没有可用的 judge 配置，CI 里跑就要联网+花钱，
  而且结果不稳定（同一个答案跑两次差 0.05 是常态）。
- 本项目的验收核心是 stale_recall_rate / refusal_rate，这两个是确定性的，
  在 metrics.py 里已经算完。faithfulness 只是辅助信号。
- 所以：本地按需跑，跑完**回填**到同一份 report JSON 里，D 的看板就能显示。

用法：
    pip install "ragas>=0.2" datasets
    python -m eval.runner.cli ragas --report report/run-xxx.json --llm <model>
"""

from __future__ import annotations

from typing import Any


class RagasUnavailable(RuntimeError):
    pass


def compute_ragas(
    samples: list[dict[str, Any]],
    *,
    llm_model: str | None = None,
    embedding_model: str | None = None,
) -> dict[str, float | None]:
    """算 faithfulness 与 answer_relevancy。

    samples 每条需含：`user_input`、`response`、`retrieved_contexts`。
    """
    try:
        from ragas import EvaluationDataset, evaluate  # type: ignore[import-not-found]
        from ragas.metrics import answer_relevancy, faithfulness  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RagasUnavailable(
            "未安装 ragas。生成类指标是可选项：pip install 'ragas>=0.2' datasets"
        ) from exc

    if not samples:
        return {"faithfulness": None, "answer_relevancy": None}

    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        raise_exceptions=False,
    )
    frame = result.to_pandas()
    # RAGAS 返回逐条分数，取均值；空列（全失败）要容错
    return {
        "faithfulness": _mean(frame.get("faithfulness")),
        "answer_relevancy": _mean(frame.get("answer_relevancy")),
    }


def compute_ragas_repeated(
    samples: list[dict[str, Any]],
    *,
    llm_model: str | None = None,
    embedding_model: str | None = None,
    repeat: int = 1,
) -> list[dict[str, float | None]]:
    """同一份样本重复跑 N 次，用于量化 judge 的**结果波动**。

    为什么必须重复：RAGAS 的 faithfulness 由 LLM judge 判定，同一答案两次跑
    差 0.05 是常态。只跑一次就把 0.87 写进汇报，等于把噪声当成事实。
    本项目的规矩是：**spread > 0.05 的那一轮结果不得用于验收**（见 runbook）。
    """
    times = max(1, int(repeat))
    return [
        compute_ragas(samples, llm_model=llm_model, embedding_model=embedding_model)
        for _ in range(times)
    ]


def summarize_runs(runs: list[dict[str, float | None]]) -> dict[str, dict[str, float | None]]:
    """把 N 次结果压成 mean / min / max / spread。

    spread = max - min，直接对应"结果波动多大"。不要用标准差：N=3 时标准差
    会给出一个看起来很专业但毫无意义的数字。
    """
    keys = sorted({k for run in runs for k in run})
    out: dict[str, dict[str, float | None]] = {}
    for key in keys:
        values = [run[key] for run in runs if isinstance(run.get(key), int | float)]
        if not values:
            out[key] = {"mean": None, "min": None, "max": None, "spread": None}
            continue
        out[key] = {
            "mean": round(sum(values) / len(values), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
            "spread": round(max(values) - min(values), 4),
        }
    return out


def build_run_record(
    *,
    report_name: str,
    created_at: str,
    llm_model: str | None,
    embedding_model: str | None,
    sample_count: int,
    repeat: int,
    duration_ms: int,
    runs: list[dict[str, float | None]],
    summary: dict[str, dict[str, float | None]],
) -> dict[str, Any]:
    """RAGAS 跑批的元数据侧车（`<run_id>.ragas.json`）。

    刻意不塞进 `run-*.json`：那份文件的键必须与 `/api/v1/eval/summary` 的 data
    段逐字一致，多一个键 D 的契约测试就会红。生成质量指标是"本地取证"，
    单独存档，D 的看板只读 metrics 里的数值。
    """
    return {
        "report": report_name,
        "created_at": created_at,
        "judge": {"llm": llm_model, "embedding": embedding_model},
        "sample_count": sample_count,
        "repeat": repeat,
        "duration_ms": duration_ms,
        "runs": runs,
        "summary": summary,
        "verdict": _verdict_text(summary),
    }


def _verdict_text(summary: dict[str, dict[str, float | None]]) -> str:
    spreads = [s["spread"] for s in summary.values() if isinstance(s.get("spread"), int | float)]
    if not spreads:
        return "无有效分数：judge 全部失败或样本为空，本轮结果不得引用"
    worst = max(spreads)
    if worst > 0.05:
        return (
            f"波动过大（spread={worst:.4f} > 0.05）：judge 不稳定或温度过高，"
            "本轮结果不得用于验收，请固定温度重跑"
        )
    return f"波动可接受（spread={worst:.4f} ≤ 0.05）"


def _mean(series: Any) -> float | None:
    if series is None:
        return None
    try:
        value = float(series.mean())
    except (TypeError, ValueError):
        return None
    return None if value != value else round(value, 4)  # NaN 检查


def to_ragas_samples(rows: list[dict], outcomes: list[Any]) -> list[dict]:
    """把我们的观测转成 RAGAS 的形状。

    `retrieved_contexts` 用不到完整正文时只放标题 —— RAGAS 只看文本相似度，
    骨架期没必要把整站正文灌进去。
    """
    by_id = {o.query_id: o for o in outcomes}
    samples = []
    for row in rows:
        o = by_id.get(row["query_id"])
        if o is None or not o.answer:
            continue
        samples.append(
            {
                "user_input": row["query"],
                "response": o.answer,
                "retrieved_contexts": [d for d in o.retrieved_doc_ids] or ["（无召回）"],
            }
        )
    return samples
