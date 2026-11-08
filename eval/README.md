# eval —— 评估跑批与报表（E 组）

回答一个问题：**这次改动让"过期版本误召回"变好了还是变坏了。**

`stale_recall_rate` 是本项目的第一指标，也是唯一能证明"时效性 RAG 真的解决了幻觉"的数字。
其余指标（faithfulness、context precision）是辅助证据。

## 快速开始

```bash
# 1. 只校验数据集，不打网络 —— CI 每次跑这个
python -m runner.cli check

# 2. 用构造数据跑一遍全部指标计算，不打网络 —— 验证指标代码没被改坏
python -m runner.cli baseline

# 3. 真跑一批（需要 C 已启动）
python -m runner.cli run --endpoint http://localhost:8080/api/v1/chat

# 4. 跑完写进 Postgres，D 的看板就能读到
# 库名是 sgu（不是 sgu_notice，sgu_notice 是 Qdrant 的 collection 名）
export DATABASE_DSN='postgres://sgu:sgu@localhost:5432/sgu?sslmode=disable'
# 也可以直接用 compose 的 canonical 变量名 POSTGRES_URL（容器内主机名是 postgres）
python -m runner.cli run --write

# 5. 卡阈值（CI 门禁用：过期误召回率超过 0.15 就失败）
python -m runner.cli run --fail-under 0.15
```

跑完得到两个文件：

- `report/run-YYYYMMDD-HHMM.json` —— 键名与 `docs/api.md` 的 `/api/v1/eval/summary` 的 `data` 段**完全一致**，
  可以直接当 Mock 响应体喂给前端
- `report/run-YYYYMMDD-HHMM.md` —— 给人看，贴 PR

## 目录

```
eval/
├── dataset/queries.jsonl     # 40 条问句：15 现行 / 10 跨年度陷阱 / 5 历史 / 10 无答案
├── runner/
│   ├── metrics.py            # 全部指标口径。纯标准库，CI 每次跑
│   ├── dataset.py            # 读 + 严格校验 queries.jsonl
│   ├── runner.py             # 走 HTTP 打被测系统，收集观测
│   ├── report.py             # 出 JSON / Markdown 两份报告
│   ├── db.py                 # 写 eval_runs / eval_results（可选，缺 psycopg 不报错）
│   ├── ragas_adapter.py      # 生成质量指标（可选，不进 CI）
│   └── cli.py                # 三个子命令：check / run / baseline
└── tests/                    # 56 例：指标口径、数据集校验、CLI 冒烟
```

## 指标口径（改之前必须读）

### stale_recall_rate —— 过期误召回率（第一指标）

```
分子 = 召回了至少一条 recency_bucket=="stale" 的 query 数
分母 = 召回了内容的、且不期待过期内容的 query 数
```

三个反直觉的地方，每一个都是踩过的坑：

1. **按 query 计数，不按 chunk 计数。** 一条问句召回了 5 条旧版本，只算 **1 个**坏样本。
   否则一条问句就能把整体指标拉爆，报表就没法看了。
2. **没召回任何东西的 query 出分母。** 它是拒答题，归 `refusal_rate` 管。
   混进来会让"到处拒答"看起来像"从不误召回"。
3. **`expect_stale_recall=true` 的题出分母。** 跨年度陷阱题（"2025 年那份通知怎么写的"）
   召回旧版本是**正确行为**。算进去会把模型训练成不敢回答任何历史问题。

### refusal_rate —— 拒答准确率

```
分子 = (该拒的拒了) + (不该拒的没拒)
分母 = expect_refusal 或 expected_doc_ids 非空的 query
```

只要一边做对就得分，所以这个指标**不能单独用**。必须和 `stale_recall_rate` 一起看：
前者高后者也高，说明是"到处拒答"；两者都高才是真的好。

### context_precision / context_recall

按 query 平均，不按 chunk 平均。无答案题的任何召回都算噪声（precision 记 0）。

### by_year

按 **query 的 `as_of`** 分组，不是按文档年份。同一份旧文档，2026 年评是过期，2025 年评就是现行。
混用会让报表失去意义 —— 这是最容易写错的地方，`tests/test_metrics.py::test_groups_by_as_of_year_not_doc_year` 盯着它。

## 为什么不用 RAGAS 算全部

| | 本文件的指标 | RAGAS |
|---|---|---|
| 时效性（本项目核心） | ✅ 有 | ❌ 没有这个概念 |
| 需要 LLM judge | ❌ | ✅ |
| 需要联网 | ❌ | ✅ |
| CI 能跑 | ✅ | ❌ |
| 结果稳定 | ✅ 确定性 | ❌ 同一答案两次差 0.05 是常态 |

所以分两层：`metrics.py` 算确定性的检索/时效指标（CI 每次跑），`ragas_adapter.py` 算生成质量（本地按需，结果回填进同一份报告）。

## 数据集规则

`dataset/queries.jsonl` 每行一条。字段：`query_id`、`query`、`as_of`、`expected_doc_ids`、
`expect_stale_recall`、`category`，可选 `expect_refusal`、`note`。

`category` 四选一：`current_rule` / `historical` / `cross_year_trap` / `no_answer`。

**硬性要求（校验会拦，`min_*` 不足即 exit 1）**：

- 总条数 ≥ 5
- `cross_year_trap` 至少 1 条 —— 没有陷阱题就没在测核心问题
- `no_answer` 至少 1 条，且必须 `expect_refusal: true` —— 不测拒答就不知道会不会硬编
- `query_id` 不能重复

**评估有效性（不足只告警，不判失败）**：陷阱题与无答案题各需 **≥10 条**（`validity_*`），
否则本轮指标不具统计意义。骨架阶段只有 4 条陷阱 + 1 条无答案，`check` 会打两条 `!` 告警；
**现已扩到 40 条（10 条陷阱 + 10 条无答案），`check` 无告警**。若日后从数据集里删题导致
任一维度跌破 10，告警会重新出现 —— 那时本轮的 `stale_recall_rate` / `refusal_rate` 就不能写进汇报。

## ragas 是可选的

```bash
pip install -e ".[ragas]"
python -m runner.cli ragas --report report/run-xxx.json --llm <model>
```

没装就报 `RagasUnavailable`，不影响任何其他命令。

## 三条硬规矩

1. **改指标口径必须同时改 `baseline` 的期望值**，并在 PR 里说明理由。往期报告要能纵向比较。
2. **不并发跑批。** 限流是 30/min，并发会把 429 混进失败率，你会以为是模型退化。
3. **不要用 `--fail-under` 卡住主干**，先把指标做出来再卡。指标没稳住就卡阈值，只会让人绕过 CI。
