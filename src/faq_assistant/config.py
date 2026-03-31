from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    milvus_host: str = Field(default="127.0.0.1", description="Хост gRPC Milvus")
    milvus_port: int = Field(default=19530, description="Порт gRPC Milvus")
    milvus_collection: str = Field(default="faq_kb", description="Базовое имя коллекции Milvus")
    milvus_user: str = Field(default="", description="Имя пользователя Milvus (необязательно)")
    milvus_password: str = Field(default="", description="Пароль Milvus (необязательно)")

    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Идентификатор модели sentence-transformers",
    )
    embedding_dimension: int = Field(
        default=384,
        ge=1,
        description="Ожидаемая размерность выхода модели эмбеддингов",
    )

    openai_api_key: str = Field(default="", description="Ключ API для чат-модели")
    openai_base_url: str | None = Field(
        default=None,
        description="Необязательный базовый URL, совместимый с OpenAI API",
    )
    llm_model: str = Field(default="gpt-4o-mini", description="Идентификатор чат-модели")

    rag_top_k: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Top-K в Milvus по категории до слияния и добора (широкий пул для нескольких источников)",
    )
    rag_min_sources: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Целевое число разных статей в контексте (ограничено top-K и размером базы)",
    )
    rag_padding_min_score: float = Field(
        default=0.22,
        ge=0.0,
        le=1.0,
        description="Ниже основного порога: включить попадание, если score≥этого значения и есть лексическое пересечение с вопросом",
    )
    llm_context_max_chunks: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Сколько лучших фрагментов после слияния и пересортировки отдать в модель и в sources",
    )
    score_threshold_no_answer: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Ниже этого сходства (1 − расстояние COSINE в Milvus) ответ не выдаём",
    )
    score_high: float = Field(default=0.55, ge=0.0, le=1.0, description="Порог уверенности «высокая»")
    score_medium: float = Field(default=0.45, ge=0.0, le=1.0, description="Порог уверенности «средняя»")

    category_route_threshold: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Минимальный косинус (вопрос+подсказка к якорю категории) для выбора коллекции",
    )

    kb_json_path: Path = Field(
        default_factory=lambda: Path("mock_kb.json"),
        description="Путь к JSON базы знаний (относительно CWD или абсолютный)",
    )
    rebuild_milvus_index: bool = Field(
        default=False,
        description="Если true — при старте удалить коллекции и переиндексировать",
    )

    log_prompts: bool = Field(
        default=True,
        description="Логировать на уровне INFO размеры промпта и ответа модели",
    )
