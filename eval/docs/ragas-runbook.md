# RAGAS 跑批手册（E 组内部）

> 为什么单独写一份：Issue #5 要求"补 RAGAS 在 CI 之外的可复现跑法"，并留下
> **用哪个 judge 模型、跑一次多少钱、结果波动多大** 的记录。这份文档就是那份记录
> 的格式与操作说明；每跑一次，把结果追加到文末的《跑批记录》表里。

## 1. 边界：RAGAS 永远不进 CI

| | `metrics.py`（确定性指标） | RAGAS（生成质量） |
|---|---|---|
| 时效性 `stale_recall_rate` | ✅ 核心指标 | ❌ 没有这个概念 |
| 需要 LLM judge | ❌ | ✅ |
| 需要联网 / 花钱 | ❌ | ✅ |
| CI 能跑 | ✅ 每次跑 | ❌ |
| 同一输入两次结果 | 完全一致 | 差 0.05 是常态 |

所以分工是：**CI 只跑 `compute()` + `baseline` 黄金用例；RAGAS 本地按需跑，结果回填进
同一份 `run-*.json` 的 `metrics`，并在旁边落一份 `<run_id>.ragas.json` 取证。**

## 2. judge 模型怎么选

优先选**中文强、便宜、支持 OpenAI 兼容接口**的模型。RAGAS 0.2 通过
`LangchainLLMWrapper(ChatOpenAI(...))` 调用，因此任何兼容 `/v1/chat/completions` 的服务都能用。

| 推荐 | 适用场景 | 备注 |
|---|---|---|
| `deepseek-chat` | 默认选择，中文判定稳、价格低 | 需 `OPENAI_API_BASE=https://api.deepseek.com/v1` |
| `qwen-plus` | 想用阿里云百炼体系时 | 需 `OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `gpt-4o-mini` | 对照实验（英文 prompt 生态最成熟） | 需要外币卡，成本高于前两者 |

**不要用**：任何带随机采样（temperature > 0）的默认配置、推理型大模型（thinking 会污染
judge 的确定性）、以及未固定版本的 `:latest` 别名——同一名字在不同时间指向不同模型，
报告就没法纵向比较了。

```bash
# 环境变量（本地 shell，不要写进 .env 提交）
export OPENAI_API_KEY=sk-xxx
export OPENAI_API_BASE=https://api.deepseek.com/v1
pip install -e ".[ragas]"     # ragas>=0.2 + datasets
```

## 3. 跑一次多少钱（量级估算）

RAGAS 每条样本大约发起 5–8 次 LLM 调用（faithfulness 拆解 1 次 + 逐句判定 2–5 次，
answer_relevancy 生成 3 个问题）。按本项目 40 条样本、平均 600 输入 token + 150 输出 token 估：

| 规模 | LLM 调用 | 输入 token | 输出 token | deepseek-chat 估算 | qwen-plus 估算 |
|---|---|---|---|---|---|
| 10 条 × 1 次（试跑） | ~60 | ~36k | ~9k | **< ¥0.2** | **< ¥0.1** |
| 40 条 × 1 次 | ~240 | ~144k | ~36k | **≈ ¥0.6** | **≈ ¥0.2** |
| 40 条 × 3 次（正式取证） | ~720 | ~432k | ~108k | **≈ ¥1.7** | **≈ ¥0.6** |

> 单价按 2026 年初官方价（deepseek-chat 输入 ¥2/M、输出 ¥8/M；qwen-plus ¥0.8/M、¥2/M）
> 估算，**以官网当期价格为准**。answer_relevancy 还会调用一次嵌入模型，费用可忽略（< ¥0.05）。
> 结论：这个项目跑一轮 RAGAS 是"一瓶水"级别，真正的成本是**你的时间**和**结果不稳**。

## 4. 波动怎么量（硬规矩）

```bash
# ① 先小样本试跑，确认 judge 通、key 有效、没有 429
python -m runner.cli ragas --report report/run-xxx.json \
  --llm deepseek-chat --embedding text-embedding-v3 --limit 10

# ② 正式取证：同一份样本跑 3 次，看 spread
python -m runner.cli ragas --report report/run-xxx.json \
  --llm deepseek-chat --embedding text-embedding-v3 --repeat 3
```

判定标准写在代码里（`ragas_adapter._verdict_text`）：

| spread = max − min | 结论 |
|---|---|
| ≤ 0.05 | 可接受，`mean` 写入报告 `metrics` |
| > 0.05 | **本轮结果不得用于验收**，检查温度 / 换 judge / 增大样本后重跑 |

命令行会直接打印 `verdict`，同时写进 `<run_id>.ragas.json` 的 `verdict` 字段——PR 里引用
的必须是 `verdict` 为"可接受"的那一轮。

## 5. 落盘产物

| 文件 | 内容 | 谁读 |
|---|---|---|
| `report/run-xxx.json` | `metrics.faithfulness` / `metrics.answer_relevancy` = 各轮均值 | D 的看板、`/api/v1/eval/summary` |
| `report/run-xxx.ragas.json` | judge 模型、样本数、repeat、每轮原始分数、mean/min/max/spread、verdict | 人（取证）、PR 附件 |

侧车刻意不塞进 `run-xxx.json`：那份 JSON 的键必须与 API 契约逐字一致，多一个键
`frontend/src/api/__tests__/contract.spec.ts` 就会红。

## 6. 跑批记录（每次追加一行）

> 现状：**尚未产生真实数据**——本机没有 judge 的 API key，`cli ragas` 目前只能验证到
> "未安装 ragas 时 exit 3"这条分支。拿到 key 后按 §4 跑，把结果填进下表，并把
> `report/*.ragas.json` 一并提交（`.gitignore` 已忽略 report 目录则贴 PR 评论）。

| 日期 | judge | 样本 | repeat | 耗时 | 费用 | faithfulness（3 次） | answer_relevancy（3 次） | spread | verdict |
|---|---|---|---|---|---|---|---|---|---|
| （待填） | deepseek-chat | 40 | 3 | — | — | — | — | — | — |

## 7. 常见故障

| 现象 | 原因 | 处置 |
|---|---|---|
| `RagasUnavailable`，exit 3 | 没装 `.[ragas]` | `pip install -e ".[ragas]"` |
| 分数全是 `None` | judge 调用全失败（key/额度/网络） | 先 `--limit 1` 跑通一条，再看异常栈 |
| 两次结果差很多 | 温度没固定 / judge 是推理模型 | 换 `deepseek-chat`，`temperature=0` |
| 429 限流 | 并发太高 | 本项目样本少，串行即可；不要给 RAGAS 加并发 |
| `answer_relevancy` 为 NaN | 嵌入模型名写错 | 显式传 `--embedding`，别依赖默认值 |
