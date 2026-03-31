from __future__ import annotations

import logging
import re
from typing import Protocol

from pydantic import BaseModel, Field

from faq_assistant.config import Settings
from faq_assistant.domain.models import ConfidenceLevel, SearchHit
from faq_assistant.ports.category_router import CategoryRouter
from faq_assistant.ports.embedding import EmbeddingProvider
from faq_assistant.ports.llm import LLMProvider
from faq_assistant.ports.vector_store import VectorStore

logger = logging.getLogger(__name__)

_NOT_FOUND = "Я не нашёл информацию по этому вопросу."

_REFUSAL_MARKERS = (
    "в контексте нет",
    "нет информации",
    "не нашёл информацию",
    "не нашла информацию",
    "не найдено информации",
    "информации нет",
    "отсутствует информация",
    "в переданном контексте ответа нет",
    "в этом контексте ответа нет",
)

_STOP_TOKENS = frozenset(
    {
        "the",
        "and",
        "for",
        "not",
        "how",
        "what",
        "это",
        "для",
        "или",
        "при",
        "все",
        "всех",
        "есть",
        "быть",
        "как",
        "что",
    }
)


def _question_keywords(question: str) -> list[str]:
    raw = [
        t.lower()
        for t in re.findall(r"[a-zа-яё0-9]+", question, re.IGNORECASE)
        if len(t) >= 3
    ]
    return [t for t in raw if t not in _STOP_TOKENS]


def _looks_like_context_refusal(text: str) -> bool:
    t = text.lower()
    return any(m in t for m in _REFUSAL_MARKERS)


def _lexical_overlap(question: str, hit: SearchHit) -> int:
    tokens = _question_keywords(question)
    if not tokens:
        return 0
    blob = f"{hit.question} {hit.answer}".lower()
    return sum(1 for tok in tokens if tok in blob)


def _best_hit_by_lexical_overlap(question: str, hits: list[SearchHit]) -> tuple[SearchHit, int]:
    best = hits[0]
    best_score = _lexical_overlap(question, best)
    for h in hits[1:]:
        s = _lexical_overlap(question, h)
        if s > best_score:
            best_score = s
            best = h
    return best, best_score


def _looks_like_leaky_context_markup(text: str) -> bool:
    """Текст похож на цитирование внутренних идентификаторов или структуры контекста, а не ответ пользователю."""
    if re.search(r"kb_\d+", text, re.IGNORECASE):
        return True
    tl = text.lower()
    if "категория=" in tl or "category=" in tl:
        return True
    if "вот блоки" in tl or "блоки, которые" in tl or "фрагменты, которые" in tl:
        return True
    return False


def _merge_and_rerank_context_hits(
    question: str,
    hits: list[SearchHit],
    threshold: float,
    *,
    lexical_weight: float = 0.09,
) -> list[SearchHit]:
    """
    Оставить попадания Milvus выше порога; добавить соседей той же категории с лексическим совпадением,
    но чуть ниже порога по score. Пересортировать так, чтобы статьи с совпадением по ключевым словам шли выше.
    """
    tokens = _question_keywords(question)
    token_n = max(len(tokens), 1)
    min_lex = 2 if token_n >= 2 else 1

    selected: dict[str, SearchHit] = {}
    for h in hits:
        if h.score >= threshold:
            selected[h.article_id] = h

    floor_relax = threshold * 0.82
    floor_deep = threshold * 0.72
    for h in hits:
        if h.article_id in selected:
            continue
        lex = _lexical_overlap(question, h)
        if lex >= min_lex and h.score >= floor_relax:
            selected[h.article_id] = h
        elif lex >= min_lex + 1 and h.score >= floor_deep:
            selected[h.article_id] = h

    merged = list(selected.values())
    merged.sort(
        key=lambda h: h.score + lexical_weight * _lexical_overlap(question, h),
        reverse=True,
    )
    return merged


def _combined_retrieval_score(question: str, hit: SearchHit, lexical_weight: float) -> float:
    return hit.score + lexical_weight * _lexical_overlap(question, hit)


def _merge_rerank_and_pad_sources(
    question: str,
    hits: list[SearchHit],
    threshold: float,
    min_sources: int,
    padding_min_score: float,
    *,
    lexical_weight: float = 0.09,
) -> list[SearchHit]:
    """
    Слияние по порогу и лексическому «спасению», затем добор из top-K Milvus до ``min_sources`` разных статей,
    чтобы в контекст попало несколько связанных FAQ, если ранжирование размазало намерение.
    Добавляются только попадания с score ≥ padding_min_score и хотя бы одним совпадением по ключевому слову.
    """
    merged = _merge_and_rerank_context_hits(
        question, hits, threshold, lexical_weight=lexical_weight
    )
    have = {h.article_id for h in merged}
    target = min(min_sources, len(hits))

    pool = sorted(
        hits,
        key=lambda h: _combined_retrieval_score(question, h, lexical_weight),
        reverse=True,
    )
    for h in pool:
        if len(merged) >= target:
            break
        if h.article_id in have:
            continue
        if h.score < padding_min_score:
            continue
        if _lexical_overlap(question, h) < 1:
            continue
        merged.append(h)
        have.add(h.article_id)

    merged.sort(
        key=lambda h: _combined_retrieval_score(question, h, lexical_weight),
        reverse=True,
    )
    return merged


class RagResult(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel


class RagService(Protocol):
    """Контракт: объект с методом ask (боевой оркестратор или заглушка в тестах)."""

    def ask(self, question: str) -> RagResult: ...


class RagOrchestrator:
    """Связывает маршрутизацию по категории, поиск по эмбеддингам и генерацию ответа."""

    def __init__(
        self,
        settings: Settings,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        llm_provider: LLMProvider,
        category_router: CategoryRouter,
    ) -> None:
        self._settings = settings
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._llm_provider = llm_provider
        self._category_router = category_router

    def ask(self, question: str) -> RagResult:
        q = question.strip()
        if not q:
            return RagResult(answer=_NOT_FOUND, sources=[], confidence=ConfidenceLevel.NONE)

        routed = self._category_router.resolve(q)
        if routed is None:
            logger.info("Категория для вопроса не найдена, символов в вопросе: %d", len(q))
            return RagResult(answer=_NOT_FOUND, sources=[], confidence=ConfidenceLevel.NONE)

        query_vectors = self._embedding_provider.embed([q])
        hits = self._vector_store.search_in_category(
            routed.category,
            query_vectors[0],
            self._settings.rag_top_k,
        )

        if not hits:
            logger.info(
                "В Milvus нет попаданий: категория=%s, символов в вопросе: %d",
                routed.category,
                len(q),
            )
            return RagResult(answer=_NOT_FOUND, sources=[], confidence=ConfidenceLevel.NONE)

        top = hits[0]
        if top.score < self._settings.score_threshold_no_answer:
            logger.info(
                "Лучший score ниже порога: категория=%s score=%.4f порог=%.4f",
                routed.category,
                top.score,
                self._settings.score_threshold_no_answer,
            )
            return RagResult(answer=_NOT_FOUND, sources=[], confidence=ConfidenceLevel.NONE)

        context_hits = _merge_rerank_and_pad_sources(
            q,
            hits,
            self._settings.score_threshold_no_answer,
            self._settings.rag_min_sources,
            self._settings.rag_padding_min_score,
        )
        max_llm = self._settings.llm_context_max_chunks
        llm_hits = context_hits[:max_llm]

        llm_out = self._llm_provider.generate_answer(q, llm_hits)
        answer_text = llm_out.text.strip()
        if answer_text and (
            _looks_like_context_refusal(answer_text) or _looks_like_leaky_context_markup(answer_text)
        ):
            best_hit, overlap = _best_hit_by_lexical_overlap(q, llm_hits)
            if overlap >= 1:
                reason = (
                    "отказ по контексту"
                    if _looks_like_context_refusal(answer_text)
                    else "утечка разметки контекста"
                )
                logger.info(
                    "Ответ модели непригоден (%s); беру ответ из базы article_id=%s совпадений_ключей=%d",
                    reason,
                    best_hit.article_id,
                    overlap,
                )
                answer_text = best_hit.answer.strip()

        sources = list(dict.fromkeys(h.article_id for h in llm_hits))
        top_context_score = max(h.score for h in llm_hits)
        confidence = self._confidence_from_score(top_context_score)

        return RagResult(answer=answer_text, sources=sources, confidence=confidence)

    def _confidence_from_score(self, top_score: float) -> ConfidenceLevel:
        if top_score >= self._settings.score_high:
            return ConfidenceLevel.HIGH
        if top_score >= self._settings.score_medium:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW
