"""Settings, read from environment variables (and a .env file if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = {"ollama": "qwen3:8b", "openai": "gpt-4o-mini", "anthropic": "claude-sonnet-5-5", "fake": "fake"}


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "ollama").lower() or "ollama")
    model: str | None = field(default_factory=lambda: os.getenv("LLM_MODEL") or None)
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    # Comma-separated: tavily, wikipedia, arxiv, fake. Default: tavily if TAVILY_API_KEY is set, else wikipedia+arxiv.
    search: str = field(default_factory=lambda: _env(
        "SEARCH_PROVIDERS", "tavily" if os.getenv("TAVILY_API_KEY") else "wikipedia,arxiv"))
    results_per_query: int = field(default_factory=lambda: int(_env("RESULTS_PER_QUERY", "4")))
    max_sub_questions: int = field(default_factory=lambda: int(_env("MAX_SUB_QUESTIONS", "4")))
    max_rounds: int = field(default_factory=lambda: int(_env("MAX_ROUNDS", "2")))  # critique -> research again
    quality_bar: int = field(default_factory=lambda: int(_env("QUALITY_BAR", "8")))  # critic score 1-10 to stop
    source_chars: int = field(default_factory=lambda: int(_env("SOURCE_CHARS", "1500")))
    approve_plan: bool = field(default_factory=lambda: _env("APPROVE_PLAN", "0") == "1")
    checkpoint_db: str = field(default_factory=lambda: _env("CHECKPOINT_DB", str(ROOT / "data" / "checkpoints.sqlite")))

    @property
    def model_name(self) -> str:
        return self.model or DEFAULT_MODELS.get(self.provider, "")


def get_settings() -> Settings:
    return Settings()
