"""E 组的命令行入口。

    python -m eval.runner.cli run       # 跑一批，出报告
    python -m eval.runner.cli check     # 只校验数据集（CI 用，不打网络）
    python -m eval.runner.cli baseline  # 用内置假数据跑一遍，验证指标代码没坏
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import dataset as dataset_mod
from . import metrics as metrics_mod
from . import report as report_mod
from .runner import Endpoint, RunnerError, run_batch

ROOT = Path(__file__).resolve().parents[1]
# 默认值可用环境变量覆盖（.env.example 与 docs/deployment.md §4 里登记的就是这两个名字）。
# 从仓库根目录跑时它们是相对路径，所以按 cwd 解析；解析不到就退回 ROOT 下的默认位置。
DEFAULT_DATASET = Path(os.environ.get("EVAL_DATASET") or ROOT / "dataset" / "queries.jsonl")
DEFAULT_REPORT_DIR = Path(os.environ.get("EVAL_OUTPUT_DIR") or ROOT / "report")


def cmd_check(args: argparse.Namespace) -> int:
    try:
        rows = dataset_mod.load(Path(args.dataset))
    except dataset_mod.DatasetError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    print(f"✓ {args.dataset}：{len(rows)} 条，类别分布 {counts}")

    # 下限来自 thresholds.json，不在这里硬编码 —— 阈值只有一处权威来源
    limits = (report_mod.thresholds().get("dataset") or {})
    need_total = int(limits.get("min_queries", 1))
    need_trap = int(limits.get("min_cross_year_trap", 1))
    need_noans = int(limits.get("min_no_answer", 1))

    problems = []
    if len(rows) < need_total:
        problems.append(f"总条数 {len(rows)} < {need_total}")
    if counts.get("cross_year_trap", 0) < need_trap:
        problems.append(f"cross_year_trap 少于 {need_trap} 条（没在测核心问题）")
    if counts.get("no_answer", 0) < need_noans:
        problems.append(f"no_answer 少于 {need_noans} 条（没在测拒答）")
    for msg in problems:
        print(f"✗ {msg}", file=sys.stderr)
    if problems:
        return 1

    # 骨架下限过了，再看评估有效性：分母太小的话指标会剧烈跳动，
    # 但这不是"数据集坏了"，所以只告警不失败 —— 骨架阶段只有 12 条，
    # 现已扩到 40 条（10 陷阱 + 10 无答案），正常情况下不再出现这两条告警。
    trap_n = counts.get("cross_year_trap", 0)
    noans_n = counts.get("no_answer", 0)
    need_trap_v = int(limits.get("validity_trap_denominator", 10))
    need_noans_v = int(limits.get("validity_no_answer_denominator", 10))
    if trap_n < need_trap_v:
        print(
            f"! 跨年度陷阱题 {trap_n} 条 < {need_trap_v}：本轮 stale_recall_rate 不具统计意义，"
            f"报告中会标注（E 的正式评估集目标 {limits.get('target_cross_year_trap', 10)} 条）",
            file=sys.stderr,
        )
    if noans_n < need_noans_v:
        print(
            f"! 无答案题 {noans_n} 条 < {need_noans_v}：本轮 refusal_rate 不具统计意义"
            f"（目标 {limits.get('target_no_answer', 10)} 条）",
            file=sys.stderr,
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    rows = dataset_mod.load(Path(args.dataset))
    endpoint = Endpoint(url=args.endpoint, token=args.token, timeout=args.timeout)
    print(f"→ {len(rows)} 条 query → {endpoint.url}")

    outcomes = run_batch(endpoint, rows)
    m = metrics_mod.compute(outcomes)

    previous = _read_previous(Path(args.report_dir))
    summary = report_mod.build_summary(outcomes, m, previous=previous)

    out_dir = Path(args.report_dir)
    json_path = report_mod.write_json(summary, out_dir, outcomes)
    md_path = report_mod.write_markdown(summary, outcomes, out_dir)
    print(f"✓ {json_path}")
    print(f"✓ {md_path}")

    for key, value in m.as_dict().items():
        print(f"  {key:22} {'—' if value is None else f'{value:.4f}'}")

    if args.write:
        from .db import DbUnavailable, write_run

        try:
            write_run(summary, outcomes)
            print("✓ 已写入 Postgres")
        except DbUnavailable as exc:
            print(f"! {exc}")

    if args.fail_under is not None:
        if m.stale_recall_rate is None or m.stale_recall_rate > args.fail_under:
            print(f"✗ stale_recall_rate 超过 {args.fail_under}", file=sys.stderr)
            return 1
    return 0


def cmd_baseline(args: argparse.Namespace) -> int:
    """不联网，用构造好的观测跑一遍全部指标计算。

    这条命令的存在意义：指标代码改坏了，CI 立刻红，不需要等真机部署。
    """
    outcomes = _baseline_outcomes()
    m = metrics_mod.compute(outcomes)
    got = {
        "stale_recall_rate": _round(m.stale_recall_rate),
        "refusal_rate": _round(m.refusal_rate),
        "context_precision": _round(m.context_precision),
        "context_recall": _round(m.context_recall),
    }
    # 手算过程（改这里之前先算一遍）：
    #   stale_recall_rate：b1 无过期（好）、b2 有过期（坏）、b3 无召回（出分母）、
    #                      b4 期望就是过期版（出分母）→ 1/2 = 0.5
    #   refusal_rate：b3 该拒且拒了，其余 3 条都该答 → 4/4 = 1.0
    #   context_precision：b1 1/2、b2 1/2、b3 无答案题不出分母、b4 1/1 → 2/3
    #   context_recall：b1 1/1、b2 1/1、b3 出分母、b4 1/1 → 3/3 = 1.0
    expected = {
        "stale_recall_rate": 0.5,
        "refusal_rate": 1.0,
        "context_precision": 0.6667,
        "context_recall": 1.0,
    }
    if got != expected:
        print(f"✗ 指标回归：期望 {expected}，实际 {got}", file=sys.stderr)
        return 1
    print(f"✓ 指标基线一致 {got}")
    return 0


def _baseline_outcomes() -> list[metrics_mod.QueryOutcome]:
    """四条构造数据，覆盖：现行命中、过期误召回、无答案正确拒答、该拒没拒。"""
    return [
        # 现行命中，没有过期版本 → 贡献 precision 1/2, recall 1/1
        metrics_mod.QueryOutcome(
            query_id="b1",
            category="current_rule",
            as_of="2026-11-08",
            expected_doc_ids=["d-2026"],
            expect_refusal=False,
            expect_stale_recall=False,
            retrieved_doc_ids=["d-2026", "d-x"],
            retrieved_buckets=["current", "current"],
        ),
        # 过期误召回 → stale_recall_rate 计入坏样本
        metrics_mod.QueryOutcome(
            query_id="b2",
            category="current_rule",
            as_of="2026-11-08",
            expected_doc_ids=["d-2026"],
            expect_refusal=False,
            expect_stale_recall=False,
            retrieved_doc_ids=["d-2025", "d-2026"],
            retrieved_buckets=["stale", "current"],
        ),
        # 无答案，正确拒答
        metrics_mod.QueryOutcome(
            query_id="b3",
            category="no_answer",
            as_of="2026-11-08",
            expected_doc_ids=[],
            expect_refusal=True,
            expect_stale_recall=False,
            refused=True,
            refuse_reason="no_hit",
        ),
        # 跨年度陷阱，期望就是过期版本 → 不计入 stale 分母
        metrics_mod.QueryOutcome(
            query_id="b4",
            category="cross_year_trap",
            as_of="2026-11-08",
            expected_doc_ids=["d-2025"],
            expect_refusal=False,
            expect_stale_recall=True,
            retrieved_doc_ids=["d-2025"],
            retrieved_buckets=["stale"],
        ),
    ]


def cmd_ragas(args: argparse.Namespace) -> int:
    """回填生成质量指标到已有报告。需要装了 ragas 且能连到 judge。

    可复现的做法（详见 eval/docs/ragas-runbook.md）：
      - `--repeat 3` 同一份样本跑三遍，看 spread 而不是单次值；
      - `--limit 10` 先小样本试跑，确认 judge 通再全量；
      - 每次都落一份 `<run_id>.ragas.json` 侧车，记录 judge 模型与每轮分数。
    """
    import time

    from . import ragas_adapter

    path = Path(args.report)
    if not path.is_file():
        print(f"✗ 报告不存在：{path}", file=sys.stderr)
        return 1
    summary = json.loads(path.read_text(encoding="utf-8"))
    # 逐条样本在旁边的 .samples.json 里，不塞进契约形状的 summary
    raw = report_mod.read_samples(path)
    samples = [
        {
            "user_input": s.get("query_id", ""),
            "response": s.get("answer") or "",
            "retrieved_contexts": s.get("retrieved_doc_ids") or ["（无召回）"],
        }
        for s in raw
        if s.get("answer")
    ]
    if args.limit is not None:
        samples = samples[: max(0, args.limit)]
    if not samples:
        print("! 报告里没有可用答案（全部拒答或 .samples.json 缺失），无可评估样本", file=sys.stderr)
        return 1

    started = time.monotonic()
    try:
        runs = ragas_adapter.compute_ragas_repeated(
            samples,
            llm_model=args.llm,
            embedding_model=args.embedding,
            repeat=args.repeat,
        )
    except ragas_adapter.RagasUnavailable as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 3
    duration_ms = int((time.monotonic() - started) * 1000)

    stats = ragas_adapter.summarize_runs(runs)
    metrics = summary.setdefault("metrics", {})
    changed = False
    for key, stat in stats.items():
        if stat["mean"] is not None:
            metrics[key] = stat["mean"]
            changed = True
    if not changed:
        print("! 没有算出任何指标（样本为空或 judge 全失败），报告未改动", file=sys.stderr)
        return 1

    record = ragas_adapter.build_run_record(
        report_name=path.name,
        created_at=report_mod.now_iso(),
        llm_model=args.llm,
        embedding_model=args.embedding,
        sample_count=len(samples),
        repeat=max(1, args.repeat),
        duration_ms=duration_ms,
        runs=runs,
        summary=stats,
    )
    meta_path = path.with_name(path.stem + ".ragas.json")
    meta_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for key, stat in stats.items():
        print(
            f"  {key:18} mean={_fmt_stat(stat['mean'])} "
            f"min={_fmt_stat(stat['min'])} max={_fmt_stat(stat['max'])} "
            f"spread={_fmt_stat(stat['spread'])}"
        )
    print(f"✓ 已回填 {len(samples)} 条样本 × {max(1, args.repeat)} 次 → {path}")
    print(f"✓ 取证记录 → {meta_path}")
    print(f"  {record['verdict']}")
    return 0


def _fmt_stat(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def _read_previous(report_dir: Path) -> dict | None:
    if not report_dir.is_dir():
        return None
    files = sorted(report_dir.glob("run-*.json"))
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _round(v: float | None) -> float | None:
    return None if v is None else round(v, 4)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eval.runner.cli", description="SGU-Notice-RAG 评估")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="校验数据集（不打网络）")
    c.add_argument("--dataset", default=str(DEFAULT_DATASET))
    c.set_defaults(func=cmd_check)

    r = sub.add_parser("run", help="跑一批评估")
    r.add_argument("--dataset", default=str(DEFAULT_DATASET))
    r.add_argument("--endpoint", default="http://localhost:8080/api/v1/chat")
    r.add_argument("--token", default=None)
    r.add_argument("--timeout", type=float, default=30.0)
    r.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    r.add_argument("--write", action="store_true", help="写入 Postgres（需要 DATABASE_DSN）")
    r.add_argument("--fail-under", type=float, default=None, help="stale_recall_rate 上限")
    r.set_defaults(func=cmd_run)

    g = sub.add_parser("ragas", help="回填生成质量指标（可选，需装 ragas）")
    g.add_argument("--report", required=True)
    g.add_argument("--llm", default=None, help="judge 模型名，如 deepseek-chat / qwen-plus")
    g.add_argument("--embedding", default=None, help="answer_relevancy 用的嵌入模型")
    g.add_argument(
        "--repeat", type=int, default=1, help="同一份样本重复跑几次（≥3 才能看出波动）"
    )
    g.add_argument("--limit", type=int, default=None, help="只评估前 N 条（先小样本试跑省成本）")
    g.set_defaults(func=cmd_ragas)

    b = sub.add_parser("baseline", help="用构造数据验证指标代码")
    b.set_defaults(func=cmd_baseline)

    args = p.parse_args(argv)
    try:
        return int(args.func(args))
    except RunnerError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 2
    except dataset_mod.DatasetError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
