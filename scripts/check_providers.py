"""模型 Provider 连通性验证（不输出任何 Key）。

用法：uv run python scripts/check_providers.py
依次验证：Embedding（维度/相似度方向性）、Rerank（相关性排序）、LLM Chat（结构化输出）。
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pydantic import BaseModel

from creditlens.common.config import get_settings
from creditlens.infrastructure.llm.chat import build_chat_provider
from creditlens.infrastructure.llm.embedding import build_embedding_provider
from creditlens.retrieval.rerank import build_reranker


class _ProbeAnswer(BaseModel):
    answer: str
    confidence: str


async def main() -> None:
    settings = get_settings()
    ok = True

    # 1. Embedding
    print(f"[1/3] Embedding provider={settings.embedding_provider} model={settings.embedding_model or '-'}")
    try:
        embedder = build_embedding_provider(settings)
        vectors = await embedder.embed_documents(
            ["流动资金贷款的准入条件", "借款人资产负债率不得高于百分之七十", "今天天气很好"]
        )
        dim = len(vectors[0])
        sim = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))  # noqa: E731
        related, unrelated = sim(vectors[0], vectors[1]), sim(vectors[0], vectors[2])
        print(f"      OK dim={dim} version={embedder.version}")
        print(f"      相关对相似度 {related:.4f} > 无关对 {unrelated:.4f}: {related > unrelated}")
    except Exception as exc:
        ok = False
        print(f"      FAILED: {type(exc).__name__}: {exc}")

    # 2. Rerank
    print(f"[2/3] Rerank provider={settings.rerank_provider} model={settings.rerank_model or '-'}")
    try:
        reranker = build_reranker(settings)
        if reranker is None:
            print("      SKIP（disabled）")
        else:
            scores = await reranker.score(
                "资产负债率的监管要求是什么",
                ["借款人资产负债率不得高于百分之七十", "贷款利率按照定价管理办法执行"],
            )
            print(f"      OK version={reranker.version} scores={[round(s, 4) for s in scores]}")
            print(f"      相关文档得分更高: {scores[0] > scores[1]}")
    except Exception as exc:
        ok = False
        print(f"      FAILED: {type(exc).__name__}: {exc}")

    # 3. LLM Chat（结构化输出）
    print(f"[3/3] LLM provider={settings.llm_provider} model={settings.llm_model or '-'}")
    try:
        chat = build_chat_provider(settings)
        if chat is None:
            print("      SKIP（disabled）")
        else:
            result = await chat.generate_structured(
                system="你是一个测试助手。",
                user="请回答：1+1 等于几？confidence 填 HIGH/MEDIUM/LOW。",
                output_schema=_ProbeAnswer,
            )
            print(f"      OK 结构化输出通过校验: answer={result.answer!r} confidence={result.confidence}")
    except Exception as exc:
        ok = False
        print(f"      FAILED: {type(exc).__name__}: {exc}")

    print("\n结论:", "全部可用 ✔" if ok else "存在失败项 ✘（见上）")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
