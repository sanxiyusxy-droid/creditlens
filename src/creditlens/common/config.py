"""应用配置。

- 密钥只通过环境变量注入，不写入仓库（文档 §18.4/§18.5）。
- RAG 与业务阈值使用版本化 YAML（config/），不全部塞进环境变量。
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "local"
    # 固定租户/用户只是本地演示身份，不是生产认证。默认要求真实
    # identity provider；本地演示必须同时显式设置 mode=demo 和 allow=true。
    api_identity_mode: str = "required"
    allow_insecure_demo_identity: bool = False

    # --- PostgreSQL ---
    database_url: str = "sqlite+aiosqlite:///./data/creditlens_local.db"

    # --- Qdrant ---
    qdrant_url: str = ":memory:"  # 本地开发允许内存模式；Compose 环境为 http://localhost:6333
    qdrant_chunks_alias: str = "credit_chunks_active"
    qdrant_summaries_alias: str = "credit_summaries_active"

    # --- 对象存储（MinIO 或本地文件系统兜底） ---
    object_store_backend: str = "local_fs"  # local_fs | minio
    local_object_root: str = str(PROJECT_ROOT / "data" / "local_objects")
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = ""
    minio_secret_key: str = ""
    minio_secure: bool = False
    minio_raw_bucket: str = "creditlens-raw"
    minio_derived_bucket: str = "creditlens-parsed"
    minio_rendered_bucket: str = "creditlens-rendered"

    # --- 模型版本（进入 Run Manifest 与 Point ID，禁止隐式漂移） ---
    embedding_provider: str = "hash_fallback"  # hash_fallback | openai_compatible
    embedding_version: str = "hash-embed-v1"
    embedding_dim: int = 256
    embedding_api_base: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""
    sparse_encoder_version: str = "bm25-jieba-v1"
    rerank_provider: str = "disabled"  # disabled | lexical_fallback | http
    rerank_api_base: str = ""
    rerank_api_key: str = ""  # 为空时回退使用 embedding_api_key（硅基流动同 Key）
    rerank_model: str = ""

    # --- 生成模型（OpenAI 兼容，如 DeepSeek） ---
    llm_provider: str = "disabled"  # disabled | openai_compatible
    llm_api_base: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    # --- Grounded QA（v1.3） ---
    qa_prompt_version: str = "grounded_qa_v1"
    qa_max_claims: int = 6
    qa_max_generation_tokens: int = 2048
    qa_max_audit_repairs: int = 1
    # 离线演示使用“原文抽取 + 引用”而非伪造模型回答；生产环境建议关闭，
    # LLM 未配置时按技术失败处理。
    qa_allow_extractive_fallback: bool = False

    @property
    def effective_embedding_version(self) -> str:
        """真实模型时以 provider+model 命名版本，避免与哈希兜底向量混用。"""
        if self.embedding_provider == "openai_compatible" and self.embedding_model:
            return f"{self.embedding_model}@api"
        return self.embedding_version

    @property
    def chunks_collection_name(self) -> str:
        """Embedding 变更 = 新物理 Collection（文档 §6.8），不覆盖旧向量。"""
        return (
            "credit_chunks_v2"
            if self.embedding_provider == "openai_compatible"
            else "credit_chunks_v1"
        )

    @property
    def summaries_collection_name(self) -> str:
        return (
            "credit_summaries_v2"
            if self.embedding_provider == "openai_compatible"
            else "credit_summaries_v1"
        )

    # --- 解析器 ---
    parser_name: str = "pymupdf"
    parser_version: str = "1.0"

    # --- 检索初始参数（文档 §8.20，评测后调整） ---
    dense_top_k: int = 30
    fused_candidate_limit: int = 80
    rrf_k: int = 60

    # --- 统一编排器（v1.1） ---
    orchestrator_enable_summary: bool = True
    orchestrator_enable_exact: bool = True
    orchestrator_enable_rerank: bool = True

    # --- Context Packing（文档 §8.13） ---
    context_token_budget: int = 4096
    context_max_per_document_ratio: float = 0.6
    context_expand_adjacent: bool = True

    # --- 上传限制（文档 §7.3） ---
    max_file_size_mb: int = 100
    max_pages: int = 500


@lru_cache
def get_settings() -> Settings:
    return Settings()
