from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pymilvus import connections

from faq_assistant.adapters.milvus_store import MilvusVectorStore
from faq_assistant.adapters.openai_llm import OpenAILLMProvider
from faq_assistant.adapters.semantic_category_router import SemanticCategoryRouter
from faq_assistant.adapters.sentence_transformers_embedding import (
    SentenceTransformersEmbeddingProvider,
)
from faq_assistant.api.routes import router
from faq_assistant.application.bootstrap import bootstrap_index
from faq_assistant.application.rag_orchestrator import RagOrchestrator, RagService
from faq_assistant.config import Settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def _disconnect_milvus_safe() -> None:
    try:
        connections.disconnect("default")
    # При остановке соединение может быть уже закрыто — игнорируем любую ошибку отключения.
    except Exception:  # noqa: BLE001
        pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    embedding_provider = SentenceTransformersEmbeddingProvider(settings)
    vector_store = MilvusVectorStore(settings, vector_dimension=embedding_provider.dimension)
    llm_provider = OpenAILLMProvider(settings)
    ctx = bootstrap_index(settings, vector_store, embedding_provider)
    category_router = SemanticCategoryRouter(
        settings,
        llm_provider,
        embedding_provider,
        ctx.category_anchors,
    )
    rag = RagOrchestrator(
        settings,
        embedding_provider,
        vector_store,
        llm_provider,
        category_router,
    )
    app.state.rag = rag
    app.state.settings = settings
    logger.info("Запуск приложения завершён")
    yield
    _disconnect_milvus_safe()
    logger.info("Остановка приложения завершена")


def create_app() -> FastAPI:
    application = FastAPI(title="TechCorp — справочник FAQ", lifespan=lifespan)
    application.include_router(router)
    return application


def create_test_app(rag: RagService) -> FastAPI:
    """Минимальное ASGI-приложение для pytest (без Milvus и загрузки модели)."""
    app = FastAPI(title="TechCorp — справочник FAQ (тест)")
    app.include_router(router)
    app.state.rag = rag
    return app


app = create_app()
