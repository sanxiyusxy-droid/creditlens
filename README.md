# CreditLens

[![CI](https://github.com/sanxiyusxy-droid/creditlens/actions/workflows/ci.yml/badge.svg)](https://github.com/sanxiyusxy-droid/creditlens/actions/workflows/ci.yml)

小微企业授信尽调与审查 Multi-Agent 系统 —— 证据优先的深度 RAG 实现。

> 本项目为技术演示，使用合成数据；系统只做"授信审查辅助与证据整理"，
> 不自动作出授信、拒贷、定价或额度决定。

## 当前版本口径

- **已发布基线**：标签 `v1.1.1` / commit `a41483d`。该版本已有公开 GitHub
  Actions 的 lint、全量非集成与真实栈 integration 三阶段绿色记录；冻结指标来自
  三份 `git_dirty=false` 报告。
- **`v1.2.0` release-ready（正式标签待创建）**：实现提交 `58f5487` + `41742cf`，
  评测源码提交 `9e64dbb`；GitHub CI #5 的 lint/unit/integration 三 Job 已成功
  （总耗时 3m14s），本地 134 项非集成与 10 项真实栈测试全绿。该版本收紧
  Auditor fail-closed、正反证据回原文、
  Evidence 跨 Run 建模、Packing `parse_run_id`、历史 Snapshot Gold 映射、Manifest
  语料 Hash、HITL 严格幂等、SSE、Outbox 并发领取、CI `pipefail`、RLS 默认业务角色
  与审计表追加写权限，并升级至 Alembic `0007_evidence_run_key`。三轮正式评测已完成；
  当前只差把报告和文档纳入最终 release commit 并创建 `v1.2.0` 标签。
  **报告记录的源码 commit 与最终 release commit 必然分离，最终 tag commit 尚不预填。**

## 架构概览

- **PostgreSQL**：业务事实源（案件、文档版本、Parse Run、Section、审计）
- **Qdrant**：可重建检索索引（Dense + BM25 Sparse，Named Vectors，Alias 蓝绿切换）
- **MinIO**：不可变原始文件与解析产物
- **FastAPI + SQLAlchemy 2 + Alembic**：API 与迁移
- **Transactional Outbox + Index Worker**：双库一致性
- 详细设计见 [CreditLens_技术实现文档.md](./CreditLens_技术实现文档.md)
- 交付进度、文档符合性对比与评测结果见 [docs/进度报告.md](./docs/进度报告.md)（每版本同步更新）

## 快速开始

### 依赖

- Python 3.12+，[uv](https://docs.astral.sh/uv/)
- Docker（可选：启动真实 PostgreSQL/Qdrant/MinIO/Redis）

### 本地离线模式（无 Docker）

默认配置使用 SQLite + Qdrant 内存模式 + 本地文件对象存储，可完整验证
上传 → 解析 → 切分 → 索引 → 检索 → 评测闭环：

```powershell
uv sync
uv run python scripts/seed_synthetic_data.py   # 合成政策 PDF -> 入库 -> 索引（三案件）
uv run python scripts/run_evaluation.py        # 默认 frozen_v2 test（121 题）五通道消融
uv run pytest -m "not integration"             # 单元 + 安全 + 非集成 E2E（CI 同一门禁）
```

> 注意：离线模式的本地库 `data/creditlens_local.db` 由 `create_all` 创建，
> **不做增量迁移**。拉取包含 Schema 变更的版本后请删除该文件重新种子；
> PostgreSQL 环境一律使用 `uv run alembic upgrade head`。

### Docker Compose 模式

```powershell
docker compose up -d
# 默认配置离线运行，应用连接使用 NOBYPASSRLS 业务角色
Copy-Item .env.example .env.local
# 迁移账号只用于建表/策略/授权，不作为 API 运行账号
$env:DATABASE_URL="postgresql+asyncpg://creditlens:creditlens@localhost:5432/creditlens"
uv run alembic upgrade head
$env:APP_DB_PASSWORD="creditlens_app"   # 仅本地演示；部署时必须替换
uv run python scripts/apply_rls.py
$env:DATABASE_URL="postgresql+asyncpg://creditlens_app:creditlens_app@localhost:5432/creditlens"
uv run python scripts/seed_synthetic_data.py
uv run uvicorn apps.api.main:app --reload
```

已有 `pg_data` 卷也可执行上述 `apply_rls.py` 补齐角色与授权；应用若连接
`creditlens` 超级用户会绕过 RLS，因此不得用该连接启动 API/Worker。

## 目录结构

```
src/creditlens/
  common/           配置、ID、哈希、时钟、错误码
  application/      Port 定义
  infrastructure/   postgres / qdrant / minio / parsers / llm Adapter
  ingestion/        上传、解析、结构切分、Outbox、Index Worker
  retrieval/        统一 Orchestrator（Dense/Sparse/Summary/Exact → RRF → Rerank → Context Packing）
  evidence/         EvidenceRef -> 原始 PDF 页 Preview
  evaluation/       GoldQuestion Schema、Recall@K/NDCG/Precision/MRR、Retrieved Evidence P/R（Refusal/Agent 指标预留答案层，不计入报告）
  agents/           Policy/Financial/Risk/Report Agent + Challenger + Auditor + Supervisor DAG
migrations/         Alembic
evaluation/datasets 冻结评测集（稳定 gold_evidence_key 锚点）
scripts/            种子数据与评测脚本
data/synthetic/     合成政策等演示数据（不含真实数据）
```

## 关键设计约束

1. 证据优先：事实性结论必须回到原文页、表格单元格或确定性计算；
2. 关系库是事实源，向量库可随时重建；
3. 时点优先：政策适用性由 `as_of_date` 决定，材料可用性由 `decision_cutoff_at` 决定；
4. 硬过滤（租户/ACL/时点/质量）在召回前执行，不允许改为降权；
5. Qdrant 命中必须回 PostgreSQL 复核，失败候选带 `rejection_reason`；
6. 评测锚点使用稳定 `gold_evidence_key`，不绑定解析生成的 UUID。

## 评测

```powershell
uv run python scripts/run_evaluation.py --dataset evaluation/datasets/frozen_v2.json --split test
```

`--split dev` 仅用于调参；简历指标只在冻结 `test` 上报。
输出 Recall@5/10/20、NDCG@K、Precision@K、MRR@10、Retrieved Evidence Precision/Recall、
per-case 指标与宏平均、独立回表 Leakage 审计（目标为 0）与 P50/P95 延迟，
报告落盘 `evaluation/reports/`，含可复现 Manifest（Git 溯源与脏工作区标记 /
dataset·uv.lock·语料 SHA256 / split / alembic revision / collection point count /
模型版本 / 每个时点组合的 snapshot_id 与内容规范化 snapshot_hash）。
口径说明：无答案生成层，不宣称 Faithfulness / Citation Accuracy / Refusal Accuracy。

数据口径（v1.1）：3 个合成案件（制造业流贷 / 科技型流贷 / 保理）、6 个逻辑文档、
8 个文档版本、97 个证据锚点、200 题（182 可答 + 18 不可答），覆盖政策 QA /
跨文档 / 财务事实 / 版本陷阱 / 口语缩写 / 拒答等意图；dev/test 按案件分层重划分，
模板组不跨 split（Leakage 自检为 0）。

### 集成测试

```powershell
# 一条命令完成：建库 → 业务角色 → Alembic → RLS 基线 → 授权 → Seed → 集成测试
# 全程使用 NOSUPERUSER NOBYPASSRLS 业务角色（不以超级用户绕过 RLS）
powershell -ExecutionPolicy Bypass -File scripts\run_integration.ps1
```

## 项目定位（诚实口径）

授信预审 RAG / Multi-Agent **在线化原型**：统一 Retrieval Orchestrator（Dense +
Sparse + Summary + Exact → RRF → Rerank → Context Packing）、中心化 Supervisor
固定 DAG 编排六项专业职责（Policy / Financial / Risk / Challenger / Auditor /
Report）、文档版本化、Snapshot 冻结、RLS 行级隔离、多案件评测集（200 题，
冻结 test 121 题）、真实栈 CI 与集成测试。
**不是生产级系统**：无真实登录/OIDC、任务队列为进程内、未做容量与压力验证；
HITL 已做行锁 + 乐观锁 + 幂等键的并发保护，但未做大规模并发压测。

## 演示（8–12 分钟）

```powershell
docker compose up -d
powershell -ExecutionPolicy Bypass -File scripts\start_demo.ps1
# 浏览器打开 http://localhost:8501
```

![演示页](docs/images/demo_screenshot.png)

五幕编排见 [docs/演示脚本.md](docs/演示脚本.md)：
① 政策时点切换（同题不同审查日 → 命中不同版本条款）→
② 完整预审 DAG（202 异步 + Claims 证据表）→ ③ 证据回原始 PDF 页 →
④ HITL 复核（blocking 全解决才 COMPLETED + 报告版本）→ ⑤ Trace 审计回放。

## v1.2.0 release-ready 冻结指标（简历口径）

评测源码 commit `9e64dbb`，frozen_v2 test split（3 案件 / 121 题 / 110 可答；
dataset SHA256 前缀 `d3dc24c3527b23f6`），bge-m3 + bge-reranker-v2-m3（API）+
bm25-jieba-v1，Alembic `0007_evidence_run_key`，真实 PG16 + Qdrant v1.18.0 +
RLS 业务角色环境下连续运行三次：

- `ablation_frozen_v2_test_20260806T150057Z.json`
- `ablation_frozen_v2_test_20260806T150712Z.json`
- `ablation_frozen_v2_test_20260806T151322Z.json`

三份 Manifest 均为 `git_dirty=false`、`git_commit=9e64dbb...`；三轮的 4 个
Snapshot Hash、`seed_script_sha256` 与真实语料 `seed_corpus_sha256` 逐位一致，
Leakage=0、unmapped=0。报告源码 commit 用于证明被测输入；包含报告与文档的最终
release commit 会是另一个 commit，待创建正式标签后再记录，不能预填。

| 通道 | Recall@10 | Recall@20 | MRR@10 | 延迟 P50/P95 |
|---|---|---|---|---|
| E0 Dense-only | 0.9545455 | 0.9681818 | 0.9484848 | 210.6–251.5 / 248.6–303.2 ms |
| E23 +Sparse+RRF | 0.9560606 | 0.9606061 | 0.9015512 | 264.3–324.1 / 392.9–472.0 ms |
| E4 Dense+Summary | 0.7833333 | 0.9681818 | 0.7622583 | 387.7–474.4 / 434.2–611.0 ms |
| E5 +QuerySpec Rewrite | 0.9515152 | 0.9606061 | 0.8711580 | 258.3–324.8 / 367.8–492.3 ms |
| **E7 全链路（默认）** | **0.9681818** | **0.9909091** | **0.8912482** | 875.8–963.6 / 1091.7–1231.6 ms |

- E0/E23/E4/E5/E7 的 Recall@10、Recall@20 与 MRR@10 三轮均逐位一致；
- E7 相比 E0：Recall@10 +1.36pp、Recall@20 +2.27pp；MRR@10 低于 Dense，说明
  全链路换取更完整的深位证据覆盖，并付出更高延迟；
- 通道决策：E7 相比 E0 有真实召回增益（R@10 +1.4pp、R@20 +2.3pp），保留为
  在线默认；E4 单独 Summary 显著弱于 Dense（R@10 0.7833 vs 0.9545），
  Summary 只作为 RRF 参与通道、不单独成链——如实取舍；
- 正式三轮以远程 bge 模型为简历主指标。离线 Hash Embedding + 词面精排基线报告
  `ablation_frozen_v2_test_20260806T145238Z.json` 的 E7 Recall@10/20 为
  **0.9469697/0.9606061**（MRR@10=0.8562374，Leakage=0、unmapped=0），用于证明
  无外部模型时的可运行性，**不作为简历主指标，也不与正式模型结果混算**。

口径说明：无答案生成层，报告指标为检索层 Recall/NDCG/MRR 与
Retrieved Evidence Precision/Recall，不宣称 Faithfulness / Citation Accuracy /
Refusal Accuracy。v1.2.0 release-ready 本地验收为 **134 passed / 10 deselected**
（非集成）以及 **10 passed / 22 deselected**（PG16 + Qdrant + Alembic 0007 +
RLS 业务角色真实栈）；GitHub CI #5 的 lint/unit/integration 三 Job 全绿（3m14s）。

## 下一版本：v1.3.0「可审计证据问答闭环」

v1.3.0 不再继续堆检索 Route，而是补齐当前缺失的答案层：Grounded QA 只消费已验证
Evidence；逐句引用审计与拒答门禁；持久化最小 Model/Tool 调用 Trace；冻结
`answer_eval_v1`（Correctness/Faithfulness/Citation/Refusal）；在 Streamlit 演示中完成
“提问 → 答案 → 引用原文 → 审计结果”的端到端展示。完成前不宣称当前版本已有
答案 Faithfulness、Citation Accuracy 或 Refusal Accuracy。


