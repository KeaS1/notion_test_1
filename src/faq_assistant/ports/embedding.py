from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Порт: преобразует пакеты текста в плотные векторы."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Размерность вектора на выходе этого провайдера."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """По одному вектору на каждый входной текст; порядок совпадает с входом."""
