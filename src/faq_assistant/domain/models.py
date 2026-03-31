from __future__ import annotations

from enum import StrEnum
from pydantic import BaseModel, Field


class KnowledgeArticle(BaseModel):
    """Одна запись FAQ из базы знаний."""

    id: str = Field(..., min_length=1)
    category: str
    question: str
    answer: str

    def embedding_text(self) -> str:
        return f"{self.question}\n{self.answer}"


class SearchHit(BaseModel):
    """Найденный фрагмент с оценкой сходства в диапазоне [0, 1]."""

    article_id: str
    category: str
    question: str
    answer: str
    score: float = Field(..., ge=0.0, le=1.0)


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class GeneratedAnswer(BaseModel):
    text: str
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM


class CategoryRouteResult(BaseModel):
    """Результат привязки вопроса к категории базы / коллекции Milvus."""

    category: str = Field(..., min_length=1)
    route_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Сходство для выбора категории (косинус по нормированным эмбеддингам)",
    )
