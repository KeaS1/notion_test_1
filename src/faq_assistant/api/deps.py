from __future__ import annotations

from fastapi import Request

from faq_assistant.application.rag_orchestrator import RagService


def get_rag_orchestrator(request: Request) -> RagService:
    rag: RagService = request.app.state.rag
    return rag
