from __future__ import annotations

from faq_assistant.domain.models import (
    CategoryRouteResult,
    ConfidenceLevel,
    GeneratedAnswer,
    KnowledgeArticle,
    SearchHit,
)
from faq_assistant.ports.category_router import CategoryRouter
from faq_assistant.ports.embedding import EmbeddingProvider
from faq_assistant.ports.llm import LLMProvider
from faq_assistant.ports.vector_store import VectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.01 * (i + 1)] * self._dimension for i in range(len(texts))]


class FakeVectorStore(VectorStore):
    def __init__(self, hits: list[SearchHit] | None = None, *, initial_count: int = 0) -> None:
        self._hits = hits or []
        self._count = initial_count
        self.upsert_calls = 0
        self.last_search_category: str | None = None

    def ensure_categories(self, categories: object) -> None:
        return None

    def drop_all_collections(self) -> None:
        self._count = 0

    def total_entities(self) -> int:
        return self._count

    def upsert_articles(self, articles: list[KnowledgeArticle], vectors: list[list[float]]) -> None:
        self.upsert_calls += 1
        self._count = len(articles)

    def search_in_category(
        self,
        category: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[SearchHit]:
        self.last_search_category = category
        return self._hits[:top_k]


class FakeLLMProvider(LLMProvider):
    def __init__(
        self,
        answer: str = "Сгенерированный ответ.",
        *,
        category_hint: str = "подсказка по теме",
    ) -> None:
        self.answer = answer
        self.category_hint = category_hint
        self.generate_calls = 0
        self.hint_calls = 0
        self.last_hits: list[SearchHit] | None = None

    def generate_answer(self, question: str, context_hits: list[SearchHit]) -> GeneratedAnswer:
        self.generate_calls += 1
        self.last_hits = context_hits
        return GeneratedAnswer(text=self.answer, confidence=ConfidenceLevel.MEDIUM)

    def infer_category_hint(self, question: str, categories: list[str]) -> str:
        self.hint_calls += 1
        return self.category_hint


class FakeCategoryRouter(CategoryRouter):
    def __init__(self, resolved: CategoryRouteResult | None) -> None:
        self._resolved = resolved

    def resolve(self, question: str) -> CategoryRouteResult | None:
        return self._resolved
