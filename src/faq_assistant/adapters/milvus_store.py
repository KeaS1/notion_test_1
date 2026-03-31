from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Iterable

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from faq_assistant.config import Settings
from faq_assistant.domain.models import KnowledgeArticle, SearchHit
from faq_assistant.ports.vector_store import VectorStore

logger = logging.getLogger(__name__)

_ALIAS = "default"
_MAX_ID = 64
_MAX_CATEGORY = 128
_MAX_QUESTION = 2048
_MAX_ANSWER = 8192


def _category_slug(category: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z_]", "_", category).lower().strip("_")
    return (s[:64] if s else "misc") or "misc"


class MilvusVectorStore(VectorStore):
    """Одна коллекция Milvus на категорию FAQ (например faq_kb_vacation)."""

    def __init__(self, settings: Settings, vector_dimension: int | None = None) -> None:
        self._settings = settings
        self._base = settings.milvus_collection
        self._dim = vector_dimension if vector_dimension is not None else settings.embedding_dimension
        self._connected = False
        self._managed_categories: set[str] = set()

    def _connect(self) -> None:
        if self._connected:
            return
        kwargs: dict[str, object] = {
            "alias": _ALIAS,
            "host": self._settings.milvus_host,
            "port": str(self._settings.milvus_port),
        }
        if self._settings.milvus_user:
            kwargs["user"] = self._settings.milvus_user
            kwargs["password"] = self._settings.milvus_password
        connections.connect(**kwargs)
        self._connected = True
        logger.info(
            "Подключение к Milvus: %s:%s",
            self._settings.milvus_host,
            self._settings.milvus_port,
        )

    def _collection_name(self, category: str) -> str:
        return f"{self._base}_{_category_slug(category)}"

    def _schema(self) -> CollectionSchema:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=_MAX_ID),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=_MAX_CATEGORY),
            FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=_MAX_QUESTION),
            FieldSchema(name="answer", dtype=DataType.VARCHAR, max_length=_MAX_ANSWER),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._dim),
        ]
        return CollectionSchema(fields=fields, description="База знаний FAQ, одна категория")

    def _create_collection(self, full_name: str) -> None:
        logger.info("Создаю коллекцию Milvus: %s", full_name)
        schema = self._schema()
        col = Collection(name=full_name, schema=schema, using=_ALIAS)
        col.create_index(
            field_name="embedding",
            index_params={"index_type": "FLAT", "metric_type": "COSINE", "params": {}},
        )
        col.load()

    def ensure_categories(self, categories: Iterable[str]) -> None:
        self._connect()
        for cat in categories:
            self._managed_categories.add(cat)
            full = self._collection_name(cat)
            if utility.has_collection(full, using=_ALIAS):
                continue
            self._create_collection(full)

    def drop_all_collections(self) -> None:
        self._connect()
        if utility.has_collection(self._base, using=_ALIAS):
            utility.drop_collection(self._base, using=_ALIAS)
            logger.info("Удалена устаревшая коллекция Milvus: %s", self._base)
        prefix = f"{self._base}_"
        for name in list(utility.list_collections(using=_ALIAS)):
            if name.startswith(prefix):
                utility.drop_collection(name, using=_ALIAS)
                logger.info("Удалена коллекция Milvus: %s", name)
        self._managed_categories.clear()

    def total_entities(self) -> int:
        self._connect()
        total = 0
        for cat in self._managed_categories:
            full = self._collection_name(cat)
            if not utility.has_collection(full, using=_ALIAS):
                continue
            col = Collection(full, using=_ALIAS)
            col.flush()
            total += int(col.num_entities)
        return total

    def upsert_articles(self, articles: list[KnowledgeArticle], vectors: list[list[float]]) -> None:
        if len(articles) != len(vectors):
            raise ValueError("Число статей и векторов должно совпадать")
        self._connect()
        cats = {a.category for a in articles}
        self.ensure_categories(cats)

        grouped: dict[str, list[tuple[KnowledgeArticle, list[float]]]] = defaultdict(list)
        for a, v in zip(articles, vectors, strict=True):
            grouped[a.category].append((a, v))

        for cat, pairs in grouped.items():
            self._upsert_one_category(cat, [p[0] for p in pairs], [p[1] for p in pairs])

        logger.info(
            "Записано статей: %d, категорий: %d",
            len(articles),
            len(grouped),
        )

    def _upsert_one_category(
        self,
        category: str,
        articles: list[KnowledgeArticle],
        vectors: list[list[float]],
    ) -> None:
        full = self._collection_name(category)
        col = Collection(full, using=_ALIAS)

        ids = [a.id for a in articles]
        categories = [a.category for a in articles]
        questions = [a.question for a in articles]
        answers = [a.answer for a in articles]

        if col.num_entities > 0:
            quoted = ",".join(f'"{i}"' for i in ids)
            col.delete(expr=f"id in [{quoted}]")

        col.insert([ids, categories, questions, answers, vectors])
        col.flush()
        col.load()

    def search_in_category(
        self,
        category: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[SearchHit]:
        self._connect()
        full = self._collection_name(category)
        if not utility.has_collection(full, using=_ALIAS):
            return []
        col = Collection(full, using=_ALIAS)
        col.load()
        if col.num_entities == 0:
            return []

        results = col.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {}},
            limit=top_k,
            output_fields=["question", "answer", "category"],
        )
        hits: list[SearchHit] = []
        for hit in results[0]:
            distance = float(hit.distance)
            similarity = 1.0 - distance
            hid = str(hit.id)
            hits.append(
                SearchHit(
                    article_id=hid,
                    category=str(hit.get("category") or ""),
                    question=str(hit.get("question") or ""),
                    answer=str(hit.get("answer") or ""),
                    score=max(0.0, min(1.0, similarity)),
                )
            )
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits
