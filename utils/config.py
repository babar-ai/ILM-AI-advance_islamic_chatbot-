
import os
from pathlib import Path
from typing import Any
from pydantic_settings import BaseSettings, SettingsConfigDict



class Settings(BaseSettings):
    VERSION: str = "1.3"
    LOGGING_DIR: str = "logs"
    
    # LLM Settings (Dual-Model Strategy)
    # Flagship model: High reasoning, pristine Arabic script, strict citation compliance
    LLM_MODEL: str = "gpt-4o"
    GROQ_LLM_MODEL: str = "openai/gpt-oss-120b"
    
    # Fast model: Ultra-fast & low cost for query classification and query rewriting
    FAST_LLM_MODEL: str = "gpt-4o-mini"
    FAST_GROQ_LLM_MODEL: str = "openai/gpt-oss-20b"

    HF_TOKEN: str = ""

    # Primary LLM Switch (Open / Close):
    # True  = Primary LLM (OpenAI) is OPEN and used first (falls back to Groq on error).
    # False = Primary LLM is CLOSED; all requests route directly to Fallback LLM (Groq).
    USE_PRIMARY_LLM: bool = True
    USE_PRIMARY_EMBEDDING: bool = True

    # Embedding Model Settings
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    FALLBACK_EMBEDDING_MODEL: str = "Orange/orange-nomic-v1.5-1536"
    EMBEDDING_DIMENSION: int = 1536


    # Qdrant Connection Settings
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""


    # Qdrant Vector Collection Names
    QURAN_COLLECTION_NAME: str = "quran"     
    HADITH_COLLECTION_NAME: str = "hadith"
    TAFSEER_COLLECTION_NAME: str = "tafsir"
    ISLAMIC_INFO_COLLECTION_NAME: str = "general_islamic_info"
    CLASSIFICATION_CACHE_COLLECTION_NAME: str = "classification_cache"

    # Hybrid Search Settings
    # SPARSE_VECTOR_NAME: the field name for BM25 sparse vectors stored in Qdrant collections.
    # Must match the name used during ingestion. Do NOT change after ingestion without re-indexing.
    SPARSE_VECTOR_NAME: str = "sparse"
    # HYBRID_SEARCH_ENABLED: when True, retrieval uses dense + sparse RRF fusion.
    # Set to False to fall back to pure semantic (dense-only) search without re-ingestion.
    HYBRID_SEARCH_ENABLED: bool = True


    # Reranking Settings
    # RERANKER_ENABLED: set to True to activate FlashRank cross-encoder reranking
    # after vector retrieval. Improves precision at the cost of ~50-150ms latency.
    RERANKER_ENABLED: bool = True
    # RERANKER_SCORE_THRESHOLD: drop documents whose FlashRank relevance score is
    # below this value. Range 0.0–1.0. Set to 0.0 to keep all reranked results.
    RERANKER_SCORE_THRESHOLD: float = 0.0


    # Redis Connection & Caching Settings (Layer 1)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""
    REDIS_TTL: int = 2592000  # 30 days in seconds (0 for no expiration)

    # Chat History Settings (LangGraph Checkpointer)
    # CHAT_HISTORY_MAX_TURNS: how many past Q&A pairs are injected into the
    # rewrite and generation nodes. Keeps context focused without token bloat.
    CHAT_HISTORY_MAX_TURNS: int = 5
    # CHAT_HISTORY_TTL: seconds before an idle session's checkpoint expires in
    # Redis. Default is 1 hour. Reset to 0 to disable expiry.
    CHAT_HISTORY_TTL: int = 3600  # 1 hour


    # Ingestion & Chunking Parameters
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 150
    BATCH_SIZE: int = 500
    FORCE_RECREATE: bool = False


    # API Keys (defaults to empty string if not present in environment or .env)
    TAVILY_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""


    # ── LangSmith Observability ───────────────────────────────────────────────
    LANGCHAIN_TRACING_V2: str = "false"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "islamic-chatbot"


    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    API_BASE_URL: str = "http://localhost:8000"

    def model_post_init(self, __context: Any) -> None:
        for alt in ("USE_MAIN_LLM", "MAIN_LLM_OPEN", "ENABLE_PRIMARY_LLM", "USE_OPENAI_LLM"):
            if alt in os.environ:
                self.USE_PRIMARY_LLM = os.environ[alt].strip().lower() in ("true", "1", "yes", "on")
                break
        for alt in ("USE_MAIN_EMBEDDING", "MAIN_EMBEDDING_OPEN", "ENABLE_PRIMARY_EMBEDDING", "USE_OPENAI_EMBEDDING"):
            if alt in os.environ:
                self.USE_PRIMARY_EMBEDDING = os.environ[alt].strip().lower() in ("true", "1", "yes", "on")
                break

        if self.HF_TOKEN:
            os.environ["HF_TOKEN"] = self.HF_TOKEN
            os.environ["HUGGING_FACE_HUB_TOKEN"] = self.HF_TOKEN
            try:
                from huggingface_hub import login
                login(token=self.HF_TOKEN, add_to_git_credential=False)
            except Exception:
                pass

    @property
    def COLLECTION_NAMES(self) -> dict[str, str]:
        return {
            "quran": self.QURAN_COLLECTION_NAME,
            "hadith": self.HADITH_COLLECTION_NAME,
            "tafseer": self.TAFSEER_COLLECTION_NAME,
            "tafsir": self.TAFSEER_COLLECTION_NAME,
            "general_islamic_info": self.ISLAMIC_INFO_COLLECTION_NAME,
        }


    @property
    def langsmith_enabled(self) -> bool:
        """Returns True if LangSmith tracing is configured and enabled."""
        return (
            self.LANGCHAIN_TRACING_V2.lower() == "true"
            and bool(self.LANGCHAIN_API_KEY)
            and self.LANGCHAIN_API_KEY != ""
        )


    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/0"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"



settings = Settings()

