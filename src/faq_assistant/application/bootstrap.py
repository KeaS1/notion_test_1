from __future__ import annotations

import logging
from dataclasses import dataclass

from faq_assistant.application.category_anchors import build_category_anchors
from faq_assistant.application.kb_loader import load_knowledge_base
from faq_assistant.config import Settings
from faq_assistant.domain.models import KnowledgeArticle
from faq_assistant.ports.embedding import EmbeddingProvider
from faq_assistant.ports.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BootstrapContext:
    articles: list[KnowledgeArticle]
    category_anchors: dict[str, str]


def bootstrap_index(
    settings: Settings,
    vector_store: VectorStore,
    embedding_provider: EmbeddingProvider,
) -> BootstrapContext:
    """Заполнить Milvus данными по категориям из kb_json_path; вернуть якоря для маршрутизации."""
    articles = load_knowledge_base(settings.kb_json_path.resolve())
    category_anchors = build_category_anchors(articles)
    categories = frozenset(a.category for a in articles)

    if settings.rebuild_milvus_index:
        logger.info("Включена полная пересборка индекса Milvus: удаляю все коллекции FAQ")
        vector_store.drop_all_collections()

    vector_store.ensure_categories(categories)

    if vector_store.total_entities() >= len(articles) and not settings.rebuild_milvus_index:
        logger.info(
            "В Milvus уже %d строк; загрузку JSON пропускаю",
            vector_store.total_entities(),
        )
        return BootstrapContext(articles=articles, category_anchors=category_anchors)

    texts = [a.embedding_text() for a in articles]
    vectors = embedding_provider.embed(texts)
    vector_store.upsert_articles(articles, vectors)
    logger.info(
        "Проиндексировано статей: %d, категорий: %d",
        len(articles),
        len(categories),
    )
    return BootstrapContext(articles=articles, category_anchors=category_anchors)
