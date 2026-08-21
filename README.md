# CreditLens

[![CI](https://github.com/sanxiyusxy-droid/creditlens/actions/workflows/ci.yml/badge.svg)](https://github.com/sanxiyusxy-droid/creditlens/actions/workflows/ci.yml)

小微企业授信尽调与审查 Multi-Agent 系统 —— 证据优先的深度 RAG 实现。

> 本项目为技术演示，使用合成数据；系统只做"授信审查辅助与证据整理"，
> 不自动作出授信、拒贷、定价或额度决定。

## 当前版本口径

- **当前开发候选**：包版本已升至 `1.5.0`，本地发布门禁已通过：Ruff check/format 与
  `uv lock --check --offline` 全绿；非集成 **487 passed / 16 skipped / 20 deselected**；真实
  PostgreSQL/Qdrant/RLS 栈 **20 passed / 22 deselected，0 skip/fail**。实现提交为 `3ed78db`；
  feature PR、GitHub CI、`main` 合并与 `v1.5.0` tag 证据仍待实际取得，不预填 run ID。
  本版将已接线的 structured Model 与 ToolGateway 四类终态统一写入
  `InvocationEnvelope v2`、追加写 `invocation_records` 和同事务 `telemetry_outbox`；新增
  at-least-once Worker（lease/reclaim/backoff/dead-letter）与 Trace 完整性复核。
- **v1.4.0 状态**：PR #2 完成评审后，`main` 以 fast-forward 合入 Head
  `7853bd2940e2240956883109e5a5133c2eab9045`；GitHub Actions main run `32440901382`
  的 lint/unit/integration 三项均成功；但 GitHub `v1.4.0` tag 尚未完成推送与 tag CI，
  因此只能称“已合入 `main`、发布闭环待完成”，不能称已发布，也不预填未知 tag run ID。
  GitLab 未推送。本版补齐失败幂等重放的稳定分类、
  结构化输出的安全 Schema
  指纹、受控分发式 Claim-Evidence Semantic Entailment 人工评审协议，以及 FULL_REVIEW Tool
  Invocation Envelope → `run_events` best-effort sink。本地 Ruff check/format、
  `uv lock --check --offline` 全绿；非集成 **433 passed / 16 skipped / 19 deselected**，其中
  16 skip 均因本机无 symlink 权限；真实 PG/Qdrant/RLS 为 **19 passed / 22 deselected**，
  0 skip、0 fail。
- **最后具有完整发布证据的版本**：标签 `v1.3.0`。第二轮被测源码为 `c4289a1`，正式预测、报告与
  发布文档由发布收口提交及 tag 承载；两类溯源职责不同，不能混写。
  已完成 Grounded QA、Evidence/Claim/Artifact 可审计闭环、业务拒答/人工复核/技术失败
  分流、模型调用脱敏 Trace、请求幂等与两阶段 gold 隔离评测。当前本地门禁为 Ruff 全绿、
  **300 项 unit/security + 19 项真实 PG/Qdrant/RLS 集成**通过。41 题第二轮正式报告见
  `evaluation/reports/answer_eval_v1_c4289a1_20260810T125108Z.json`。GitHub PR #1 的
  CI run `31599090053` 与合并后 main run `31599459084` 均为 lint/unit/integration
  三阶段全绿。
- **更早发布版本**：`v1.2.0` 的实现提交为 `58f5487` + `41742cf`，评测源码提交为
  `9e64dbb`；其冻结检索正式指标与历史证据继续保留。

## 架构概览

- **PostgreSQL**：业务事实源（案件、文档版本、Parse Run、Section、审计）
- **Qdrant**：可重建检索索引（Dense + BM25 Sparse，Named Vectors，Alias 蓝绿切换）
- **MinIO**：不可变原始文件与解析产物
- **FastAPI + SQLAlchemy 2 + Alembic**：API 与迁移
- **Transactional Outbox + Index Worker**：双库一致性
- **Invocation Ledger + Telemetry Outbox**：Model/Tool 脱敏调用事实、可靠投递与 Trace 完整性
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
  observability/    InvocationEnvelope v2、追加写调用账本、Telemetry Outbox Worker
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
6. 评测锚点使用稳定 `gold_evidence_key`，不绑定解析生成的 UUID；
7. 已接线 Model/Tool 调用先提交脱敏事实与投递意图，外部导出采用 at-least-once，
   不宣称 exactly-once。

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
冻结 test 121 题）、真实栈 CI 与集成测试；v1.5 候选新增 append-only Model/Tool 调用账本、
Transactional Telemetry Outbox 与可复核 Trace。
**不是生产级系统**：无真实登录/OIDC、任务队列为进程内、未做容量与压力验证；
HITL 已做行锁 + 乐观锁 + 幂等键的并发保护，但未做大规模并发压测；Telemetry Worker
默认关闭，专用生产 Worker role 尚未实现；正式部署仍需专用 Exporter/Worker 身份、指标
平台和告警闭环。

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

## v1.2.0 已发布冻结指标（简历口径）

评测源码 commit `9e64dbb`，frozen_v2 test split（3 案件 / 121 题 / 110 可答；
dataset SHA256 前缀 `d3dc24c3527b23f6`），bge-m3 + bge-reranker-v2-m3（API）+
bm25-jieba-v1，Alembic `0007_evidence_run_key`，真实 PG16 + Qdrant v1.18.0 +
RLS 业务角色环境下连续运行三次：

- `ablation_frozen_v2_test_20260806T150057Z.json`
- `ablation_frozen_v2_test_20260806T150712Z.json`
- `ablation_frozen_v2_test_20260806T151322Z.json`

三份 Manifest 均为 `git_dirty=false`、`git_commit=9e64dbb...`；三轮的 4 个
Snapshot Hash、`seed_script_sha256` 与真实语料 `seed_corpus_sha256` 逐位一致，
Leakage=0、unmapped=0。报告源码 commit 用于证明被测输入；标签 `v1.2.0` 承载报告、
文档与发布口径，因此它与报告 Manifest 中的评测源码 commit 分离。

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
Refusal Accuracy。v1.2.0 本地发布验收为 **134 passed / 10 deselected**
（非集成）以及 **10 passed / 22 deselected**（PG16 + Qdrant + Alembic 0007 +
RLS 业务角色真实栈）；最终候选 GitHub CI #6 的 lint/unit/integration 三 Job 全绿
（4m05s）。

## v1.3.0：可审计证据问答闭环

v1.3.0 补齐答案层：Grounded QA 只消费已验证 Evidence，模型只生成受限 Claim 草稿；
服务端生成权威 ID/状态并执行 Evidence 白名单、数字、阈值方向、明显极性、Locator/Hash/
Snapshot 门禁；结果分为 `ANSWERED / ABSTAINED / NEEDS_REVIEW / TECHNICAL_FAILURE`，
技术故障不得伪装成业务拒答。Streamlit 已支持“问题 → 答案状态 → 逐句引用 → 原文预览 →
审计 Trace”。

`answer_eval_v1` 使用 3 个合成案件、41 题（30 可答、11 不可答）。第二轮正式状态为
**23 ANSWERED / 8 REFUSED / 6 NEEDS_REVIEW / 4 TECHNICAL_FAILURE**；确定性
Lexical Correctness **16.67%**、Key-point Recall **23.64%**、Numeric Accuracy
**20.59%**、Citation-set P/R/F1 **76.67%/69.70%/73.02%**、Refusal Accuracy
**72.73%**、False Refusal **0**、Technical Failure **9.76%**。相较首轮，技术失败率
由 17.07% 降至 9.76%，确定性通过率由 19.51% 升至 31.71%。这些是合成集上的
`DETERMINISTIC_LEXICAL_AND_CITATION_SET` 指标，`semantic_entailment_evaluated=false`；
**Citation F1 不是 Faithfulness，Lexical Correctness 也不是语义准确率**。剩余 4 个技术
失败为 1 个数字引用审计失败和 3 个连续 Schema 校验失败，作为当前已知限制保留。

## v1.4.0：失败可解释、受控语义评审与调用观测

- 新请求使用版本化 `grounded_qa_request_v2`；失败幂等重放只接受唯一、最后且与
  Run/tenant/case/状态转换一致的固定错误枚举。v1.3 已完成请求可在旧 Hash 与原始创建
  事件同时匹配时兼容重放，旧失败和伪装降级 fail closed；执行取消会保留
  `CancelledError` 控制流；`shield` 只保护 best-effort 失败收口，并未设置应用层 timeout。
  DB 可用时通常以 `QA_CALL_CANCELLED` 终结 Run；DB 同时故障时不以清理异常覆盖原取消，
  仍需生产级 lease/reconciler 收口。
- 从正式 41 题 prediction 与 gold-free checkpoint 生成可审计语义 Source：**41 Claim /
  46 supporting evidence**，Source SHA-256 为 `b77c6d8f...6cfb5e5`。完整工具链为
  `Source → 两套高熵伪名 package/worksheet → 两名隔离真人 → compile submission →`
  `争议项盲化 adjudication package/worksheet → 第三名隔离真人 → compile →`
  `score --require-complete`。这里“三名真人”由私有 roster、隔离账号和人工监督保证；代码
  只能校验两个不同 reviewer 伪名、不同于它们的 adjudicator 伪名及 HUMAN 自我声明，不能
  证明三个不同物理真人。当前 41 Claim 的技术门禁要求 82 个 rating，且所有非严格共识项均
  被裁决为最终标签或 `EXCLUDE`。两套 review package 显式共用同一带时区
  `generated-at`；裁决 package 也显式冻结不早于两份 review `submitted_at` 的独立 aware
  `generated-at`。固定各自绑定的输入、私有伪名、seed 与时间可逐字节复现产物。代码只
  校验伪名为 22–128 位 URL-safe 字符；CSPRNG 高熵与不复用属于管理员操作协议。
- 公开裁决包只暴露匿名随机顺序的 prior decision，不含 rationale、question/claim ID、评审
  身份或 gold，但 package/worksheet 仍只受控分发；
  Source、所有 package/worksheet、review/adjudication mapping、身份 roster、submission 和报告
  均不进入 Git。真人隔离 attestation 是自我声明，不是技术证明。**当前已私有生成两套各
  41 项的 package + blank worksheet + mapping kit，但尚无真人 label、submission 或
  adjudication**；`formal-readiness-incomplete.json` 仅诊断当前为 INCOMPLETE、收到 0/82
  rating，不是正式语义报告或分数，不能引用其中的零值 label rate。未来 final entailed rate
  也只是 Claim 语义支持代理指标，不是答案正确率，更不能称为 Faithfulness。
- ToolGateway 记录 `SUCCESS/FAILED/DENIED/CANCELLED` 统一信封，输入输出使用实例密钥
  HMAC 指纹；同任务预算在调用前原子占位，Run 级 ContextVar 隔离并发事件。FULL_REVIEW
  只 best-effort 写 `run_events`：普通 sink/DB 失败不反转工具业务结果，但对应 RunEvent 可能
  缺失，只有内存 record 的 `observability_error_codes` 留痕；sink 取消会继续传播。模型侧仍
  直接持久化 legacy `MODEL_INVOCATION_*`；`adapt_model_invocation_trace()` 只是显式转换
  helper，生产 QA 路径尚未自动接入统一 envelope/writer。当前也没有 durable telemetry
  outbox/retry 或完整 OTel backend；成本只有在显式版本化价格表和完整 Token 下才估算，
  不是供应商账单。

执行协议与诚实边界见
[v1.4 受控语义评审与调用观测](docs/v1.4_语义盲审与调用观测.md)。v1.3 报告仍保持
`semantic_entailment_evaluated=false`；Citation-set F1 不能因为评测协议就绪而改称
Faithfulness。

## v1.5.0 候选：持久调用账本与遥测投递

- `InvocationEnvelope` 升级为 `invocation_v2`。Grounded QA、Policy Agent 的已接线
  structured Model 调用，以及 ToolGateway 的 `SUCCESS/FAILED/DENIED/CANCELLED`，统一
  进入独立短审计事务；`invocation_records` 与 `telemetry_outbox` 同事务提交。v2 sink
  取代对应 legacy 调用事件，不双写；`generate_text` 等自由文本调用仍不在覆盖范围。
  v1.5 的取消 timeout 只限制调用方等待 shield 事务的时间，不是 DB deadline；等待超时后
  底层事务仍可能稍后提交或失败。v1.4 的历史取消路径没有这项调用方 timeout。
- `invocation_records` 是 append-only 调用事实；同 Invocation UUID、同 Hash、同父绑定可
  幂等重放，冲突内容或缺失 Outbox fail closed。Tool 参数/结果只保存实例密钥 HMAC 指纹，
  不保存 Prompt、原始参数、模型原文、异常正文或 traceback。
- Outbox Worker 使用 at-least-once 投递、Invocation UUID 幂等键、PostgreSQL
  `SKIP LOCKED`、lease/reclaim、指数退避和 `DEAD`。Exporter 取消留下的过期 lease 在次数
  耗尽后转 `DEAD`，不会无限重试。`TelemetryDelivery` 显式携带
  tenant/case/run/invocation/topic。
- `GET /runs/{id}/trace` 返回 invocations 与投递汇总，并复核 payload、Hash、关系列投影、
  Outbox 和 `review_runs` 绑定；汇总明确区分 `PENDING/COMPLETE/DEGRADED/EMPTY`，历史未
  回填 Run 为 `LEGACY_UNAVAILABLE`。`MISSING` 是合法 InvocationRecord 缺对应 Outbox；
  `INVALID` 是 payload/Hash/投影/绑定损坏；无效持久 payload 不回显。`COMPLETE` 只证明
  当前持久 Invocation 集合有效且均已投递，不证明所有实际调用都已入账。`EMPTY` 既可能是
  合法零调用，也可能是提交前丢失；没有 expected-invocation set/reservation/count 时无法区分。
- Worker 默认 disabled；内置 Noop Exporter fail closed，且只允许 local/dev/test 的显式
  单租户生命周期验证。当前列级 Outbox `UPDATE` 仍由 API/内置 Worker 共用的
  `creditlens_app` 持有，数据库不能区分二者；专用生产 Worker role 尚未实现。生产需要专用
  幂等 Exporter、独立 Worker 和 service role/tenant shard 身份。本版不是 exactly-once，
  也未提供 OTel、Prometheus、Dashboard、历史回填、Run
  lease/heartbeat/reconciler 或持久任务队列；数据库事务提交前故障仍不能保证留痕。

完整设计、故障矩阵、部署与发布前检查见
[v1.5 持久调用账本与遥测投递](docs/v1.5_持久调用账本与遥测投递.md)。当前状态是通过
本地门禁的发布候选；PR/CI、`main` 与 tag 证据将在实际取得后更新。


