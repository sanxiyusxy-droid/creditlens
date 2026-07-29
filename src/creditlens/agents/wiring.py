"""Multi-Agent 装配：注册工具、授权 Allowlist、组装 Supervisor。"""

from qdrant_client import QdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from creditlens.agents.auditor import EvidenceAuditor
from creditlens.agents.challenger import Challenger
from creditlens.agents.financial_agent import FinancialAgent
from creditlens.agents.policy_agent import PolicyAgent
from creditlens.agents.supervisor import Supervisor
from creditlens.application.ports import EmbeddingProvider
from creditlens.application.snapshot_service import SnapshotContext
from creditlens.formulas.engine import FormulaRegistry
from creditlens.retrieval.contracts import TrustedRequestContext
from creditlens.retrieval.hybrid import HybridRetriever
from creditlens.tools.finance_tools import compute_metric_for_entity
from creditlens.tools.gateway import ToolGateway


def build_supervisor(
    session: AsyncSession,
    qdrant: QdrantClient,
    embedder: EmbeddingProvider,
    snapshot: SnapshotContext,
    registry: FormulaRegistry | None = None,
    chat=None,
) -> tuple[Supervisor, ToolGateway]:
    """组装工具与 Agent。检索工具绑定 Snapshot 冻结的物理 Collection 与
    Parse Run 集合（文档 §6.4：Run 不读"当前 Active"）。session 由调用方管理事务。

    chat：OpenAICompatChat 实例；提供时 Policy Agent 的 statement 由 LLM 概括
    （失败自动降级模板），证据绑定与 Contract 校验不变。"""
    registry = registry or FormulaRegistry()
    retriever = HybridRetriever(qdrant, embedder)
    gateway = ToolGateway()
    chunks_collection = snapshot.chunks_collection

    async def search_policy(trusted: TrustedRequestContext, query: str):
        return await retriever.retrieve(
            session, trusted, query, chunks_collection,
            final_limit=8, enable_rerank=False, snapshot=snapshot,
        )

    async def search_counter_evidence(trusted: TrustedRequestContext, query: str):
        return await retriever.retrieve(
            session, trusted, query, chunks_collection,
            final_limit=5, enable_rerank=False, snapshot=snapshot,
        )

    async def compute_metric(
        trusted: TrustedRequestContext, metric_code: str, formula_version: str, period_end
    ):
        return await compute_metric_for_entity(
            session,
            registry,
            tenant_id=trusted.tenant_id,
            entity_id=trusted.borrower_entity_id,
            metric_code=metric_code,
            formula_version=formula_version,
            period_end=period_end,
            # P0-1：案件范围 + 审查时点 + Snapshot 冻结事实集合
            case_id=trusted.case_id,
            decision_cutoff_at=trusted.decision_cutoff_at,
            allowed_fact_ids=snapshot.allowed_fact_ids,
        )

    gateway.register("search_policy", search_policy)
    gateway.register("search_counter_evidence", search_counter_evidence)
    gateway.register("compute_metric", compute_metric)

    # Capability Allowlist（文档 §10.4）：Policy 不许算财务，Financial 不许检索政策
    gateway.grant("policy_analyst", ["search_policy"])
    gateway.grant("financial_analyst", ["compute_metric"])
    gateway.grant("challenger", ["search_counter_evidence"])

    supervisor = Supervisor(
        policy_agent=PolicyAgent(gateway, chat=chat),
        financial_agent=FinancialAgent(gateway),
        challenger=Challenger(gateway),
        auditor=EvidenceAuditor(registry),
    )
    return supervisor, gateway
