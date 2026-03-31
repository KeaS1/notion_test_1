from __future__ import annotations

import logging

from sentence_transformers import SentenceTransformer

from faq_assistant.config import Settings
from faq_assistant.ports.embedding import EmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = SentenceTransformer(settings.embedding_model_name)
        inferred = int(self._model.get_sentence_embedding_dimension())
        if inferred != settings.embedding_dimension:
            logger.warning(
                "embedding_dimension=%s не совпадает с размерностью модели=%s; использую размерность модели",
                settings.embedding_dimension,
                inferred,
            )
        self._dimension = inferred

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [row.tolist() for row in vectors]
