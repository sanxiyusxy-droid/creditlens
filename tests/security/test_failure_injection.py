"""故障注入测试（任务 28 收尾，文档 §13/§11.4：降级不得假成功）。

覆盖：
- LLM 不可用：Policy Agent statement 降级模板，Claim/证据契约不变；
- LLM 输出违反输出边界（批准/拒贷措辞）：强制降级模板（§12.7）；
- Reranker 不可用：降级 RRF 顺序，channel_config 记录 rerank_degraded=true；
- Embedding 不可用：检索直接失败（抛异常），不返回空结果假装成功；
- Qdrant 不可用：同上，异常向上传播。
"""

import uuid

import pytest

from creditlens.agents.policy_agent import PolicyAgent
from creditlens.infrastructure.llm.embedding import HashEmbedding
from creditlens.retrieval.contracts import TrustedRequestContext
from creditlens.retrieval.dense import DenseRetriever
from creditlens.retrieval.hybrid import HybridRetriever
from creditlens.tools.gateway import ToolGateway
from tests.e2e.test_ingest_retrieve_e2e import TENANT_ID, seeded  # noqa: F401  复用夹具

COLLECTION = "credit_chunks_v1"


async def _trusted(session, case_id) -> TrustedRequestContext:
    from creditlens.application.trusted_context import build_trusted_context

    return await build_trusted_context(session, TENANT_ID, case_id)


class _BrokenChat:
    """模拟 LLM API 超时/故障。"""

    async def generate_structured(self, **kwargs):
        raise TimeoutError("simulated LLM timeout")


class _DecisiveChat:
    """模拟 LLM 输出越界决策性措辞。"""

    async def generate_structured(self, output_schema, **kwargs):
        return output_schema(statement="该企业满足条件，建议批准贷款并尽快放款执行。")


class _BrokenReranker:
    version = "broken-v1"

    async def score(self, query, documents):
        raise ConnectionError("simulated rerank service down")


class _BrokenEmbedding(HashEmbedding):
    async def embed_query(self, text):
        raise ConnectionError("simulated embedding API down")


class _BrokenQdrant:
    def query_points(self, *args, **kwargs):
        raise ConnectionError("simulated qdrant down")


def _policy_gateway(session, qdrant, embedder, snapshot) -> ToolGateway:
    from creditlens.retrieval.hybrid import HybridRetriever as HR

    retriever = HR(qdrant, embedder)
    gateway = ToolGateway()

    async def search_policy(trusted, query):
        return await retriever.retrieve(
            session, trusted, query, COLLECTION,
            final_limit=8, enable_rerank=False, snapshot=snapshot,
        )

    gateway.register("search_policy", search_policy)
    gateway.grant("policy_analyst", ["search_policy"])
    return gateway


async def _prepare(session, seeded):  # noqa: F811
    from creditlens.application.snapshot_service import freeze_snapshot

    trusted = await _trusted(session, seeded["case_id"])
    snapshot = await freeze_snapshot(session, trusted, chunks_collection=COLLECTION)
    return trusted, snapshot


async def test_llm_timeout_degrades_to_template(session, qdrant, seeded):  # noqa: F811
    """LLM 超时：statement 降级模板；证据绑定与 verdict 不受影响。"""
    trusted, snapshot = await _prepare(session, seeded)
    gateway = _policy_gateway(session, qdrant, seeded["embedder"], snapshot)
    agent = PolicyAgent(gateway, chat=_BrokenChat())
    artifact = await agent.run(uuid.uuid4(), "policy_review", trusted)

    supported = [c for c in artifact.claims if c.verdict == "SUPPORTED"]
    assert supported, "LLM 故障不应导致 Claim 丢失"
    assert all("审查日适用政策中与「" in c.statement for c in supported), "应为降级模板语句"
    assert all(c.supporting_evidence_ids for c in supported)


async def test_llm_decisive_output_forced_to_template(session, qdrant, seeded):  # noqa: F811
    """LLM 输出"批准贷款"类措辞：输出边界强制降级（§12.7）。"""
    trusted, snapshot = await _prepare(session, seeded)
    gateway = _policy_gateway(session, qdrant, seeded["embedder"], snapshot)
    agent = PolicyAgent(gateway, chat=_DecisiveChat())
    artifact = await agent.run(uuid.uuid4(), "policy_review", trusted)

    for claim in artifact.claims:
        assert "批准" not in claim.statement
        assert "放款" not in claim.statement


async def test_reranker_down_degrades_to_rrf_and_records(session, qdrant, seeded):  # noqa: F811
    """Reranker 宕机：结果按 RRF 顺序返回，rerank_degraded 必须记录。"""
    trusted, snapshot = await _prepare(session, seeded)
    retriever = HybridRetriever(qdrant, seeded["embedder"], reranker=_BrokenReranker())
    result = await retriever.retrieve(
        session, trusted, "资产负债率要求", COLLECTION,
        final_limit=8, enable_rerank=True, snapshot=snapshot,
    )
    assert result.candidates, "降级后仍应返回 RRF 结果"
    assert result.channel_config["rerank"] is False
    assert result.channel_config["rerank_degraded"] is True


async def test_embedding_down_fails_loudly(session, qdrant, seeded):  # noqa: F811
    """Embedding API 宕机：必须抛异常，不得返回空结果假装成功。"""
    trusted, snapshot = await _prepare(session, seeded)
    retriever = DenseRetriever(qdrant, _BrokenEmbedding())
    with pytest.raises(ConnectionError):
        await retriever.retrieve(
            session, trusted, "资产负债率", COLLECTION, top_k=5, snapshot=snapshot
        )


async def test_qdrant_down_fails_loudly(session, seeded):  # noqa: F811
    """Qdrant 宕机：必须抛异常，不得假成功。"""
    trusted = await _trusted(session, seeded["case_id"])
    retriever = DenseRetriever(_BrokenQdrant(), seeded["embedder"])
    with pytest.raises(ConnectionError):
        await retriever.retrieve(session, trusted, "资产负债率", COLLECTION, top_k=5)
