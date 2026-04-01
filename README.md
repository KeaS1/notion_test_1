# TechCorp — справочник FAQ (RAG, Milvus)

HTTP API для ответов на вопросы сотрудников по мок-базе знаний (`mock_kb.json`): **маршрутизация по категории**, семантический поиск в **Milvus** (отдельная коллекция на каждую категорию из JSON) и генерация ответа через **OpenAI Chat Completions**. Стек: Python 3.11 (локально и в Docker-образе приложения), FastAPI, Pydantic v2, pytest.

## Архитектура

Слои и **порты (ABC)**:

- **Domain** — `KnowledgeArticle`, `SearchHit`, `GeneratedAnswer`, `ConfidenceLevel`, `CategoryRouteResult`.
- **Ports** — `EmbeddingProvider`, `VectorStore`, `LLMProvider`, `CategoryRouter` (`src/faq_assistant/ports/`).
- **Adapters** — `SentenceTransformersEmbeddingProvider`, `MilvusVectorStore` (коллекции `{MILVUS_COLLECTION}_{category}`), `OpenAILLMProvider`, `SemanticCategoryRouter`.
- **Поток вопроса** — `LLMProvider.infer_category_hint` → эмбеддинг `вопрос + подсказка` → косинус к якорям категорий (тексты из примеров вопросов по категории) → выбор коллекции → эмбеддинг **исходного** вопроса → `search_in_category` в Milvus → при достаточной релевантности `generate_answer`.
- **Application** — `bootstrap_index` (возвращает `BootstrapContext` с якорями), `RagOrchestrator`.
- **API** — `POST /api/ask`, Pydantic-схемы, `RagService` (Protocol) для подмены в тестах.

Новые стратегии маршрутизации — отдельная реализация `CategoryRouter`; новое хранилище — реализация `VectorStore`.

## Запуск локально

1. Поднять Milvus (например через Docker Compose из этого репозитория, только сервисы `etcd`, `minio`, `milvus`):

   ```bash
   docker compose up -d etcd minio milvus
   ```

2. Создать виртуальное окружение и установить зависимости:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -e ".[dev]"
   ```

3. Скопировать `.env.example` в `.env`, задать `OPENAI_API_KEY`.

4. Из корня репозитория (чтобы находился `mock_kb.json`):

   ```bash
   uvicorn faq_assistant.main:app --reload --app-dir src
   ```

## Запуск через Docker Compose (Milvus + приложение)

```bash
set OPENAI_API_KEY=sk-...
docker compose up --build
```

Первый старт приложения скачает модель эмбеддингов (sentence-transformers); индексация в Milvus выполняется при старте, если суммарное число строк по всем категориям меньше размера KB. Полная пересборка: `REBUILD_MILVUS_INDEX=true` (удаляет старую монолитную коллекцию `MILVUS_COLLECTION`, если осталась от прежней версии, и все `{MILVUS_COLLECTION}_*`).

## Пример запроса

```bash
curl -s -X POST http://127.0.0.1:8000/api/ask ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"Как запросить отпуск?\"}"
```

Ответ:

```json
{
  "answer": "...",
  "sources": ["kb_002"],
  "confidence": "high"
}
```

Поле `confidence`: `high` / `medium` / `low` по порогам similarity топ-1; `none`, если релевантных документов нет (текст: «Я не нашёл информацию по этому вопросу.»).

## Тесты

```bash
.venv\Scripts\python -m pytest tests -v
```

Тесты не требуют Milvus и OpenAI: используются фейковые реализации портов и лёгкое тестовое приложение без lifespan.

## Переменные окружения

См. [.env.example](.env.example). Важные:

| Переменная | Описание |
|------------|----------|
| `MILVUS_HOST`, `MILVUS_PORT` | Адрес gRPC Milvus |
| `OPENAI_API_KEY` | Ключ API для чат-модели |
| `LLM_MODEL` | Модель чата (по умолчанию `gpt-4o-mini`) |
| `EMBEDDING_MODEL_NAME` | Модель sentence-transformers |
| `EMBEDDING_DIMENSION` | Ожидаемая размерность (384 для `all-MiniLM-L6-v2`; при расхождении с моделью в лог пишется предупреждение, в Milvus используется размерность модели) |
| `RAG_TOP_K`, `SCORE_THRESHOLD_NO_ANSWER`, `SCORE_HIGH`, `SCORE_MEDIUM` | Поведение RAG |
| `CATEGORY_ROUTE_THRESHOLD` | Порог косинуса для выбора Milvus-коллекции по якорю категории |
| `KB_JSON_PATH` | Путь к JSON базы знаний |
| `REBUILD_MILVUS_INDEX` | `true` — удалить коллекцию и переиндексировать при старте |
| `LOG_PROMPTS` | Логировать размер промпта и ответа чат-модели |

## Pre-commit

```bash
pip install pre-commit
pre-commit install
```

## Допущения

- В Milvus в индекс попадает текст `question + "\n" + answer` для каждой статьи; статьи одной категории — в одной коллекции.
- Маршрутизация категории: сначала ответ LLM (короткая подсказка по теме), затем семантическое сравнение с якорями категорий на тех же эмбеддингах, что и для FAQ.
- Метрика поиска: **COSINE**; в ответ API similarity = `1 - distance` из Milvus (ограничено [0, 1]).
- Индекс Milvus: **FLAT** (достаточно для малой базы).
- При `total_entities() >= len(KB)` и `REBUILD_MILVUS_INDEX=false` повторная загрузка JSON при старте **пропускается** (для обновления данных используйте `REBUILD_MILVUS_INDEX=true` или очистку volume).

## OpenAPI

После запуска сервера: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).
