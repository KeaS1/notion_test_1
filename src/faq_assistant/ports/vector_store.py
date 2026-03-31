from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable

from faq_assistant.domain.models import KnowledgeArticle, SearchHit


class VectorStore(ABC):
    """Порт: коллекции Milvus по категориям и поиск по сходству внутри одной категории."""

    @abstractmethod
    def ensure_categories(self, categories: Iterable[str]) -> None:
        """Создать коллекции для каждого идентификатора категории, если их ещё нет."""

    @abstractmethod
    def drop_all_collections(self) -> None:
        """Удалить все коллекции FAQ, которыми управляет это хранилище."""

    @abstractmethod
    def total_entities(self) -> int:
        """Сумма числа строк по всем управляемым коллекциям категорий."""

    @abstractmethod
    def upsert_articles(self, articles: list[KnowledgeArticle], vectors: list[list[float]]) -> None:
        """Заменить или вставить строки по категориям; векторы соответствуют статьям."""

    @abstractmethod
    def search_in_category(
        self,
        category: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[SearchHit]:
        """Семантический поиск только в коллекции для указанной ``category``."""
