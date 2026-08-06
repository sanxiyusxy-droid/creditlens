# CreditLens

[![CI](https://github.com/sanxiyusxy-droid/creditlens/actions/workflows/ci.yml/badge.svg)](https://github.com/sanxiyusxy-droid/creditlens/actions/workflows/ci.yml)

小微企业授信尽调与审查 Multi-Agent 系统 —— 证据优先的深度 RAG 实现。

> 本项目为技术演示，使用合成数据；系统只做"授信审查辅助与证据整理"，
> 不自动作出授信、拒贷、定价或额度决定。

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
Copy-Item .env.example .env   # 按需填写 MinIO 凭据等
$env:DATABASE_URL="postgresql+asyncpg://creditlens:creditlens@localhost:5432/creditlens"
uv run alembic upgrade head
uv run uvicorn apps.api.main:app --reload
```

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

## 冻结指标（v1.1.1，简历口径）

frozen_v2 冻结 test split（3 案件 / 121 题 / 110 可答；数据集 sha256 前缀
`d3dc24c3527b23f6`），bge-m3 + bge-reranker-v2-m3（API）+ bm25-jieba-v1，
Alembic `0006_hitl_wp3`，真实栈（PG16 + Qdrant v1.18.0）+ RLS 业务角色、
一次性干净环境（全新数据库与 Collection，幂等 Seed）下，
**三次连续评测**（报告 `evaluation/reports/ablation_frozen_v2_test_
{20260806T131128Z, 20260806T131616Z, 20260806T132057Z}.json`，
Manifest `git_dirty=false`、三轮的 4 个冻结世界 snapshot_hash 逐位一致）：

| 通道 | Recall@10 | Recall@20 | MRR@10 | 延迟 P50/P95 |
|---|---|---|---|---|
| E0 Dense-only | 0.9545 | 0.9682 | 0.9485 | 207–229 / 237–302 ms |
| E23 +Sparse+RRF | 0.9561 | 0.9606 | 0.8970 | 248–258 / 336–345 ms |
| E4 Dense+Summary | 0.7788 | 0.9682 | 0.7572 | 383–403 / 437–489 ms |
| E5 +QuerySpec Rewrite | 0.9515 | 0.9606 | 0.8666 | 247–260 / 329–395 ms |
| **E7 全链路（默认）** | **0.9682** | **0.9909** | 0.8912 | 822–828 / 965–1026 ms |

- E0/E23/E4/E5 全部指标与 E7 Recall@10/@20/MRR@10 三次**逐位一致**；
  独立回表 Leakage = **0**（全通道）、unmapped = 0、时点/ACL 拒绝为 0；
- E7 per-case Recall@10：case001 0.9490 / case002 0.9881 / case003 0.9737，
  宏平均 0.9703；
- 与 v1.1 数字的关系：主链（E0/E23/E5/E7）聚合指标逐位不变；E4 由 0.7697
  变为 0.7788（±1 题）——v1.1 环境集合含早期实验残留点（chunks 125 vs
  干净环境 250），v1.1.1 按规约在一次性干净环境重冻，以本节数字为准；
- 通道决策：E7 相比 E0 有真实召回增益（R@10 +1.4pp、R@20 +2.3pp），保留为
  在线默认；E4 单独 Summary 显著弱于 Dense（R@10 0.7788 vs 0.9545），
  Summary 只作为 RRF 参与通道、不单独成链——如实取舍；
- 诚实口径：E7 的 @5 系列与 NDCG 存在 ±1 题的运行间波动（case003，
  远程 rerank API 侧分数抖动；融合/精排层已做确定性 tie-break，
  Recall@10/@20 全稳），简历只引用 Recall@10/@20 与 Leakage。

口径说明：无答案生成层，报告指标为检索层 Recall/NDCG/MRR 与
Retrieved Evidence Precision/Recall，不宣称 Faithfulness / Citation Accuracy /
Refusal Accuracy。自动化测试：103 passed（非集成，0 skip）+ 9 项真实栈
集成测试（含 3 项 HITL 并发，0 skip / 0 fail）。


