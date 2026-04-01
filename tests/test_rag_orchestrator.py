from __future__ import annotations

from faq_assistant.application.rag_orchestrator import RagOrchestrator
from faq_assistant.config import Settings
from faq_assistant.domain.models import CategoryRouteResult, SearchHit
from tests.fakes import FakeCategoryRouter, FakeEmbeddingProvider, FakeLLMProvider, FakeVectorStore


def _hit(article_id: str, score: float, category: str = "vacation") -> SearchHit:
    return SearchHit(
        article_id=article_id,
        category=category,
        question="q",
        answer="a",
        score=score,
    )


def test_ask_returns_not_found_when_route_fails() -> None:
    settings = Settings()
    rag = RagOrchestrator(
        settings,
        FakeEmbeddingProvider(),
        FakeVectorStore(hits=[]),
        FakeLLMProvider(),
        FakeCategoryRouter(None),
    )
    out = rag.ask("что-то")
    assert "не нашёл" in out.answer.lower()
    assert out.sources == []
    assert out.confidence.value == "none"


def test_ask_returns_not_found_when_no_hits_in_category() -> None:
    settings = Settings()
    rag = RagOrchestrator(
        settings,
        FakeEmbeddingProvider(),
        FakeVectorStore(hits=[]),
        FakeLLMProvider(),
        FakeCategoryRouter(CategoryRouteResult(category="vacation", route_score=0.9)),
    )
    out = rag.ask("что-то")
    assert "не нашёл" in out.answer.lower()


def test_ask_returns_not_found_when_top_score_below_threshold() -> None:
    settings = Settings(score_threshold_no_answer=0.9)
    rag = RagOrchestrator(
        settings,
        FakeEmbeddingProvider(),
        FakeVectorStore(hits=[_hit("kb_001", 0.2)]),
        FakeLLMProvider(),
        FakeCategoryRouter(CategoryRouteResult(category="vacation", route_score=0.5)),
    )
    out = rag.ask("вопрос")
    assert "не нашёл" in out.answer.lower()
    assert out.sources == []


def test_ask_fallback_when_llm_claims_no_context_but_article_matches() -> None:
    """При отказе по контексту подставляем ответ статьи с лучшим лексическим пересечением."""
    settings = Settings(score_threshold_no_answer=0.3)
    hits = [
        SearchHit(
            article_id="kb_005",
            category="equipment",
            question="Как заказать оборудование?",
            answer="Оборудование заказывается через Slack-канал #it-equipment.",
            score=0.7,
        ),
        SearchHit(
            article_id="kb_006",
            category="equipment",
            question="Какой ноутбук выдают разработчикам?",
            answer="Разработчики получают MacBook Pro 14\" или аналог на выбор.",
            score=0.65,
        ),
    ]
    llm = FakeLLMProvider(answer="В контексте нет информации о доступных ноутбуках.")
    rag = RagOrchestrator(
        settings,
        FakeEmbeddingProvider(),
        FakeVectorStore(hits=hits),
        llm,
        FakeCategoryRouter(CategoryRouteResult(category="equipment", route_score=0.8)),
    )
    out = rag.ask("Какой ноутбук выдают разработчикам?")
    assert "MacBook" in out.answer
    assert set(out.sources) == {"kb_005", "kb_006"}
    assert llm.generate_calls == 1


def test_ask_rescues_lexical_match_below_threshold_and_replaces_meta_answer() -> None:
    """Тянем релевантную статью чуть ниже порога по score, если есть совпадение по словам; убираем «мета»-ответ."""
    settings = Settings(
        score_threshold_no_answer=0.35,
        score_high=0.55,
        score_medium=0.45,
    )
    hits = [
        SearchHit(
            article_id="kb_006",
            category="equipment",
            question="Какой ноутбук выдают разработчикам?",
            answer="Разработчики получают MacBook.",
            score=0.52,
        ),
        SearchHit(
            article_id="kb_005",
            category="equipment",
            question="Как заказать оборудование?",
            answer="Оборудование заказывается через Slack-канал #it-equipment.",
            score=0.30,
        ),
    ]
    llm = FakeLLMProvider(
        answer='Вот блоки, которые подходят:\n\n1. kb_006 category=equipment\n\nответ: тест.',
    )
    rag = RagOrchestrator(
        settings,
        FakeEmbeddingProvider(),
        FakeVectorStore(hits=hits),
        llm,
        FakeCategoryRouter(CategoryRouteResult(category="equipment", route_score=0.8)),
    )
    out = rag.ask("я хочу заказать оборудование как мне это сделать")
    assert "#it-equipment" in out.answer or "Slack" in out.answer
    assert "kb_006" not in out.answer
    assert set(out.sources) == {"kb_005", "kb_006"}


def test_ask_pads_second_source_when_semantic_top_is_narrow() -> None:
    """Добор второй статьи из top-K при узком семантическом топе и лексическом пересечении."""
    settings = Settings(
        score_threshold_no_answer=0.35,
        score_high=0.55,
        score_medium=0.45,
        rag_min_sources=3,
        rag_padding_min_score=0.22,
        llm_context_max_chunks=3,
    )
    llm = FakeLLMProvider(answer="ok")
    hits = [
        SearchHit(
            article_id="kb_006",
            category="equipment",
            question="Какой ноутбук выдают разработчикам?",
            answer="MacBook.",
            score=0.52,
        ),
        SearchHit(
            article_id="kb_005",
            category="equipment",
            question="Как заказать оборудование?",
            answer="Через Slack #it-equipment.",
            score=0.24,
        ),
    ]
    rag = RagOrchestrator(
        settings,
        FakeEmbeddingProvider(),
        FakeVectorStore(hits=hits),
        llm,
        FakeCategoryRouter(CategoryRouteResult(category="equipment", route_score=0.8)),
    )
    out = rag.ask("заказать оборудование и какой ноутбук")
    assert len(out.sources) >= 2
    assert "kb_005" in out.sources
    assert "kb_006" in out.sources
    assert llm.last_hits is not None
    assert len(llm.last_hits) >= 2


def test_ask_passes_only_llm_context_max_chunks_to_llm() -> None:
    """В модель уходит не больше llm_context_max_chunks фрагментов после ранжирования."""
    settings = Settings(
        score_threshold_no_answer=0.25,
        score_high=0.5,
        score_medium=0.4,
        llm_context_max_chunks=3,
    )
    llm = FakeLLMProvider(answer="ok")
    hits = [_hit(f"kb_{i:03d}", 0.9 - i * 0.01) for i in range(1, 6)]
    rag = RagOrchestrator(
        settings,
        FakeEmbeddingProvider(),
        FakeVectorStore(hits=hits),
        llm,
        FakeCategoryRouter(CategoryRouteResult(category="vacation", route_score=0.8)),
    )
    rag.ask("вопрос")
    assert llm.last_hits is not None
    assert len(llm.last_hits) == 3
    assert [h.article_id for h in llm.last_hits] == ["kb_001", "kb_002", "kb_003"]


def test_ask_calls_llm_and_sets_sources_when_relevant() -> None:
    settings = Settings(
        score_threshold_no_answer=0.3,
        score_high=0.5,
        score_medium=0.4,
    )
    llm = FakeLLMProvider(answer="Ответ из контекста.")
    hits = [
        _hit("kb_002", 0.8),
        _hit("kb_003", 0.4),
    ]
    store = FakeVectorStore(hits=hits)
    rag = RagOrchestrator(
        settings,
        FakeEmbeddingProvider(),
        store,
        llm,
        FakeCategoryRouter(CategoryRouteResult(category="vacation", route_score=0.8)),
    )
    out = rag.ask("Как запросить отпуск?")
    assert out.answer == "Ответ из контекста."
    assert out.sources == ["kb_002", "kb_003"]
    assert out.confidence.value == "high"
    assert llm.generate_calls == 1
    assert llm.last_hits is not None
    assert len(llm.last_hits) == 2
    assert store.last_search_category == "vacation"


def test_ask_strips_whitespace_question() -> None:
    settings = Settings()
    rag = RagOrchestrator(
        settings,
        FakeEmbeddingProvider(),
        FakeVectorStore(hits=[]),
        FakeLLMProvider(),
        FakeCategoryRouter(None),
    )
    out = rag.ask("   ")
    assert "не нашёл" in out.answer.lower()
