from __future__ import annotations

import json
from pathlib import Path

from faq_assistant.domain.models import KnowledgeArticle


def load_knowledge_base(path: Path) -> list[KnowledgeArticle]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON базы знаний должен быть списком объектов")
    return [KnowledgeArticle.model_validate(item) for item in data]
