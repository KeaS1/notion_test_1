from __future__ import annotations

import json
from pathlib import Path

from faq_assistant.application.bootstrap import bootstrap_index
from faq_assistant.config import Settings
from tests.fakes import FakeEmbeddingProvider, FakeVectorStore


def _write_kb(path: Path, n: int = 2) -> None:
    rows: list[dict[str, str]] = []
    for i in range(n):
        rows.append(
            {
                "id": f"kb_{i:03d}",
                "category": "c",
                "question": f"q{i}",
                "answer": f"a{i}",
            }
        )
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")


def test_bootstrap_ingests_when_store_empty(tmp_path: Path) -> None:
    kb = tmp_path / "kb.json"
    _write_kb(kb, n=2)
    settings = Settings(kb_json_path=kb, rebuild_milvus_index=False)
    embed = FakeEmbeddingProvider(dimension=8)
    store = FakeVectorStore(initial_count=0)
    bootstrap_index(settings, store, embed)
    assert store.upsert_calls == 1


def test_bootstrap_skips_when_collection_already_full(tmp_path: Path) -> None:
    kb = tmp_path / "kb.json"
    _write_kb(kb, n=3)
    settings = Settings(kb_json_path=kb, rebuild_milvus_index=False)
    embed = FakeEmbeddingProvider()
    store = FakeVectorStore(initial_count=10)
    bootstrap_index(settings, store, embed)
    assert store.upsert_calls == 0
