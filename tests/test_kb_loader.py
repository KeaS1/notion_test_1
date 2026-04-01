from __future__ import annotations

import json
from pathlib import Path

import pytest

from faq_assistant.application.kb_loader import load_knowledge_base


def test_load_knowledge_base_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "kb.json"
    data = [
        {
            "id": "kb_001",
            "category": "x",
            "question": "Q?",
            "answer": "A.",
        }
    ]
    path.write_text(json.dumps(data), encoding="utf-8")
    articles = load_knowledge_base(path)
    assert len(articles) == 1
    assert articles[0].id == "kb_001"
    assert "Q?" in articles[0].embedding_text()


def test_load_knowledge_base_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / "kb.json"
    path.write_text(json.dumps({"bad": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="списком"):
        load_knowledge_base(path)
