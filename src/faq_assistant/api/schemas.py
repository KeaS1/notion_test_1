from pydantic import BaseModel, Field

from faq_assistant.domain.models import ConfidenceLevel


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Вопрос сотрудника по внутреннему FAQ")


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: ConfidenceLevel
