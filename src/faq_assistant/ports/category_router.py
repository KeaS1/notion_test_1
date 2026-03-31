from __future__ import annotations

from abc import ABC, abstractmethod

from faq_assistant.domain.models import CategoryRouteResult


class CategoryRouter(ABC):
    """Порт: сопоставить вопрос пользователя с канонической категорией базы (ключ коллекции)."""

    @abstractmethod
    def resolve(self, question: str) -> CategoryRouteResult | None:
        """Вернуть лучшую категорию при достаточной уверенности, иначе None."""
