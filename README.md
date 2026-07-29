# CreditLens

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
uv run python scripts/seed_synthetic_data.py   # 合成政策 PDF -> 入库 -> 索引
uv run python scripts/run_evaluation.py        # 20 条 Smoke 问题 Recall@K 报告
uv run pytest                                  # 单元 + 端到端测试
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
  retrieval/        检索契约、硬过滤、Dense Retriever（后续：Sparse/RRF/Rerank）
  evidence/         EvidenceRef -> 原始 PDF 页 Preview
  evaluation/       GoldQuestion Schema 与 Recall@K
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
uv run python scripts/run_evaluation.py --dataset evaluation/datasets/frozen_v1.json
```

输出 Recall@5/10/20、MRR@10、AllRequiredEvidence@K、独立回表 Leakage 审计
（目标为 0）与 P50/P95 延迟，报告落盘 `evaluation/reports/`，含可复现 Manifest
（dataset hash / alembic revision / collection point count / 模型版本）。

数据口径（如实声明）：1 个合成案件、3 个逻辑文档、4 个文档版本（政策新旧两版 +
监管指引 + 年报节选）、80 题（74 可答 + 6 拒答，拒答题不计入 Recall）。
拒答正确率 / Unsupported Answer 属答案生成层指标，当前系统无自由问答生成层，
待该层实现后补充。

## 项目定位（诚实口径）

授信预审 RAG / Multi-Agent **原型**：深度 RAG 实验链路与中心化 Multi-Agent
主流程完整可运行，含文档版本化、混合召回消融、证据定位、财务公式重放、
Snapshot 冻结（文档 + 财务事实）、RLS 行级隔离与 60 项自动化测试。
**不是生产级系统**：无真实登录/OIDC、任务队列为进程内、未做并发与容量验证。

