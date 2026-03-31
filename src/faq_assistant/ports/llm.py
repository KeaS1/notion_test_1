from __future__ import annotations

from abc import ABC, abstractmethod

from faq_assistant.domain.models import GeneratedAnswer, SearchHit


class LLMProvider(ABC):
    """Порт: формирует ответ на естественном языке по найденному контексту."""

    @abstractmethod
    def generate_answer(self, question: str, context_hits: list[SearchHit]) -> GeneratedAnswer:
        """Опираться только на переданные фрагменты как на фактический контекст."""

    @abstractmethod
    def infer_category_hint(self, question: str, categories: list[str]) -> str:
        """Короткая текстовая подсказка по области вопроса (для маршрутизации по категории)."""
