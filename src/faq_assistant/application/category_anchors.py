from __future__ import annotations

from faq_assistant.domain.models import KnowledgeArticle


def build_category_anchors(
    articles: list[KnowledgeArticle],
    *,
    max_sample_questions: int = 6,
) -> dict[str, str]:
    """Собрать текстовый якорь маршрутизации по категории из примеров вопросов в базе."""
    by_cat: dict[str, list[KnowledgeArticle]] = {}
    for a in articles:
        by_cat.setdefault(a.category, []).append(a)
    anchors: dict[str, str] = {}
    for cat, group in by_cat.items():
        samples = [x.question for x in group[:max_sample_questions]]
        joined = "; ".join(samples)
        anchors[cat] = f"Категория {cat}. Типичные вопросы сотрудников: {joined}"
    return anchors
