from __future__ import annotations

from fastapi import APIRouter, Depends

from faq_assistant.api.deps import get_rag_orchestrator
from faq_assistant.api.schemas import AskRequest, AskResponse
from faq_assistant.application.rag_orchestrator import RagService

router = APIRouter(tags=["Справочник"])


@router.post("/api/ask", response_model=AskResponse)
def ask_question(
    body: AskRequest,
    rag: RagService = Depends(get_rag_orchestrator),
) -> AskResponse:
    result = rag.ask(body.question)
    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        confidence=result.confidence,
    )
