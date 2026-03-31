from __future__ import annotations

import logging
import math

from faq_assistant.config import Settings
from faq_assistant.domain.models import CategoryRouteResult
from faq_assistant.ports.category_router import CategoryRouter
from faq_assistant.ports.embedding import EmbeddingProvider
from faq_assistant.ports.llm import LLMProvider

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SemanticCategoryRouter(CategoryRouter):
    """
    1) Модель выдаёт короткую подсказку по теме вопроса.
    2) Эмбеддинг ``вопрос + подсказка`` сравнивается с заранее посчитанными эмбеддингами якорей категорий.
    """

    def __init__(
        self,
        settings: Settings,
        llm: LLMProvider,
        embedding: EmbeddingProvider,
        category_anchors: dict[str, str],
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._embedding = embedding
        self._anchors = dict(category_anchors)
        self._categories = sorted(self._anchors.keys())
        self._anchor_vectors: dict[str, list[float]] = {}
        for cat in self._categories:
            self._anchor_vectors[cat] = self._embedding.embed([self._anchors[cat]])[0]

    def resolve(self, question: str) -> CategoryRouteResult | None:
        if not self._categories:
            return None
        hint = self._llm.infer_category_hint(question, self._categories)
        combined = f"{question}\n{hint}"
        query_vec = self._embedding.embed([combined])[0]

        best_cat: str | None = None
        best_score = -1.0
        for cat, anchor_vec in self._anchor_vectors.items():
            score = _cosine_similarity(query_vec, anchor_vec)
            if score > best_score:
                best_score = score
                best_cat = cat

        if best_cat is None or best_score < self._settings.category_route_threshold:
            logger.info(
                "Маршрутизация не удалась: лучшая=%s score=%.4f порог=%.4f",
                best_cat,
                best_score,
                self._settings.category_route_threshold,
            )
            return None

        logger.info(
            "Вопрос отнесён к категории=%s score_маршрута=%.4f символов_в_подсказке=%d",
            best_cat,
            best_score,
            len(hint),
        )
        return CategoryRouteResult(category=best_cat, route_score=min(1.0, max(0.0, best_score)))
