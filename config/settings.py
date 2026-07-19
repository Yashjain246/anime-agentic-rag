"""
config/settings.py
──────────────────
Central configuration using pydantic-settings.
All values can be overridden via environment variables or the .env file.
"""

from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR_DEFAULT = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── API Keys ──────────────────────────────────────────────────────────────
    GOOGLE_API_KEY: str = Field(..., description="Gemini API key from aistudio.google.com")
    TAVILY_API_KEY: str = Field(..., description="Tavily search API key")
    OMDB_API_KEY: str = Field(..., description="OMDB API key for episode ratings")

    # ── LangSmith (optional — set to enable tracing) ──────────────────────────
    LANGSMITH_TRACING: bool = Field(default=False)
    LANGSMITH_API_KEY: str = Field(default="", description="LangSmith API key")
    LANGSMITH_PROJECT: str = Field(default="anime-rag-prod")
    LANGSMITH_ENDPOINT: str = Field(default="https://api.smith.langchain.com")

    # ── Data paths ────────────────────────────────────────────────────────────
    DATA_DIR: Path = Field(default=DATA_DIR_DEFAULT)

    @property
    def CHROMA_LORE_DIR(self) -> Path:
        return self.DATA_DIR / "chroma_anime_db"

    @property
    def CHROMA_RECS_DIR(self) -> Path:
        return self.DATA_DIR / "chroma_recs_db"

    @property
    def CHROMA_LORE_ZIP(self) -> Path:
        return self.DATA_DIR / "chroma_anime_db.zip"

    @property
    def CHROMA_RECS_ZIP(self) -> Path:
        return self.DATA_DIR / "chroma_recs_db.zip"

    @property
    def CHARACTER_DB_PATH(self) -> Path:
        return self.DATA_DIR / "all_characters (4).jsonl"

    @property
    def MANGA_CHAPTERS_PATH(self) -> Path:
        return self.DATA_DIR / "manga_chapters (3).jsonl"

    @property
    def EPISODE_MAPPING_PATH(self) -> Path:
        return self.DATA_DIR / "episode_mapping (1).jsonl"

    @property
    def CHARTS_DIR(self) -> Path:
        return PROJECT_ROOT / "charts"

    # ── Model configuration ───────────────────────────────────────────────────
    LORE_EMBEDDING_MODEL: str = Field(default="BAAI/bge-small-en-v1.5")
    RECS_EMBEDDING_MODEL: str = Field(default="BAAI/bge-small-en-v1.5")
    RERANKER_MODEL: str = Field(default="ms-marco-MiniLM-L-12-v2")
    LLM_MODEL: str = Field(default="gemini-3.1-flash-lite")
    QUERY_GEN_TEMPERATURE: float = Field(default=0.0)
    AGENT_TEMPERATURE: float = Field(default=0.7)

    # ── Agent behaviour ───────────────────────────────────────────────────────
    MAX_TOOL_ITERATIONS: int = Field(
        default=5,
        description="Maximum tool calls per request — prevents infinite loops",
    )
    BM25_TOP_K: int = Field(default=12)
    DENSE_TOP_K: int = Field(default=12)
    RERANKER_TOP_N: int = Field(default=3)
    SCORE_THRESHOLD: float = Field(default=0.95)
    RECS_TOP_K: int = Field(default=5)

    # ── Calendar tool ─────────────────────────────────────────────────────────
    ENABLE_CALENDAR_TOOL: bool = Field(
        default=False,
        description="Enable Google Calendar tool. Requires token.json from one-time OAuth.",
    )
    CALENDAR_TOKEN_PATH: Path = Field(default=PROJECT_ROOT / "token.json")
    CALENDAR_CREDENTIALS_PATH: Path = Field(default=PROJECT_ROOT / "credentials.json")
    CALENDAR_TOKEN_B64: str = Field(default="", description="Base64 encoded token for calendar auth in production.")

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str = Field(
        default=f"sqlite:///{PROJECT_ROOT / 'chat_history.db'}",
        description="SQLite locally, PostgreSQL on production (Supabase)",
    )

    # ── Admin panel ───────────────────────────────────────────────────────────
    ADMIN_PASSWORD: str = Field(
        default="",
        description="Password to unlock the admin panel (DB stats + clear-all) "
                    "in the Streamlit sidebar. Leave empty to disable the panel entirely.",
    )

    def setup_langsmith(self) -> None:
        """Inject LangSmith env vars so LangGraph picks them up automatically."""
        if self.LANGSMITH_TRACING and self.LANGSMITH_API_KEY:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.LANGSMITH_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.LANGSMITH_PROJECT
            os.environ["LANGCHAIN_ENDPOINT"] = self.LANGSMITH_ENDPOINT

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        self.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Singleton ─────────────────────────────────────────────────────────────────
settings = Settings()
