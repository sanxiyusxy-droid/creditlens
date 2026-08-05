"""版本化 BM25 稀疏编码器（任务 14，文档 §6.6/§8.7 Route B）。

- jieba 固定分词 + 停用词表 + 词项哈希 -> Qdrant SparseVector 索引；
- 文档端只写 TF（词频），IDF 由 Qdrant `Modifier.IDF` 在检索时计算，
  语料统计随物理 Collection 冻结，符合"IDF Snapshot 随索引版本"要求；
- 编码器版本（分词器+停用词+哈希方案）进入 sparse_encoder_version，
  不同版本不能写入同一 Collection。
"""

import hashlib
from collections import Counter

import jieba

SPARSE_ENCODER_NAME = "bm25-jieba-v1"

# 精简中文停用词表（版本化：改动即升级 encoder version）
_STOPWORDS = frozenset(
    [
        "的",
        "了",
        "和",
        "是",
        "在",
        "有",
        "与",
        "及",
        "或",
        "对",
        "为",
        "由",
        "于",
        "从",
        "被",
        "把",
        "就",
        "都",
        "而",
        "且",
        "其",
        "之",
        "以",
        "等",
        "各",
        "该",
        "本",
        "我",
        "你",
        "他",
        "她",
        "它",
        "们",
    ]
)


def _term_id(token: str) -> int:
    """词项 -> 稳定 32 位索引（特征哈希，避免维护全局词表）。"""
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def tokenize(text: str) -> list[str]:
    return [
        token.strip()
        for token in jieba.cut_for_search(text)
        if token.strip() and token.strip() not in _STOPWORDS and len(token.strip()) > 1
    ]


class Bm25SparseEncoder:
    """输出 (indices, values)；文档端 value=TF，查询端 value=1。"""

    version = SPARSE_ENCODER_NAME

    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        counts = Counter(_term_id(t) for t in tokenize(text))
        if not counts:
            return [], []
        indices = sorted(counts)
        values = [float(counts[i]) for i in indices]
        return indices, values

    def encode_query(
        self, text: str, extra_terms: list[str] | None = None
    ) -> tuple[list[int], list[float]]:
        tokens = tokenize(text)
        for term in extra_terms or []:
            tokens.extend(tokenize(term))
            tokens.append(term)  # 精确术语整词也进入查询
        ids = {_term_id(t) for t in tokens if t}
        indices = sorted(ids)
        return indices, [1.0] * len(indices)
