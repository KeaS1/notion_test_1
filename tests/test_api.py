from __future__ import annotations

from fastapi.testclient import TestClient

from faq_assistant.application.rag_orchestrator import RagResult
from faq_assistant.domain.models import ConfidenceLevel
from faq_assistant.main import create_test_app


class DummyRag:
    def __init__(self, result: RagResult) -> None:
        self._result = result

    def ask(self, question: str) -> RagResult:
        return self._result


def test_api_ask_returns_payload() -> None:
    result = RagResult(
        answer="Тестовый ответ",
        sources=["kb_001"],
        confidence=ConfidenceLevel.HIGH,
    )
    app = create_test_app(DummyRag(result))
    client = TestClient(app)
    res = client.post("/api/ask", json={"question": "Какой график?"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "Тестовый ответ"
    assert body["sources"] == ["kb_001"]
    assert body["confidence"] == "high"


def test_api_ask_validation_error_on_empty_question() -> None:
    result = RagResult(answer="x", sources=[], confidence=ConfidenceLevel.NONE)
    app = create_test_app(DummyRag(result))
    client = TestClient(app)
    res = client.post("/api/ask", json={"question": ""})
    assert res.status_code == 422
