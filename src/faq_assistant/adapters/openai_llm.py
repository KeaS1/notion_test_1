from __future__ import annotations

import logging

from openai import OpenAI

from faq_assistant.config import Settings
from faq_assistant.domain.models import ConfidenceLevel, GeneratedAnswer, SearchHit
from faq_assistant.ports.llm import LLMProvider

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Ты дружелюбный внутренний помощник сотрудников TechCorp: говори тепло и по-человечески, но по делу, без лишней "
    "воды и без канцелярита ради канцелярита. Ниже — короткий список самых релевантных выдержек из базы знаний; у каждой Q: "
    "(типичный вопрос) и A: (официальный ответ). Ответь сотруднику так, будто пишешь в корпоративный чат: можно "
    "коротко признать вопрос («понимаю», «да, это у нас так»), дальше ясно передай факты из подходящего A: — "
    "перефразировать можно, смысл и цифры не искажай и ничего не выдумывай. Не перечисляй номера фрагментов, не "
    "цитируй внутренние id, не описывай структуру контекста. Если ни одна выдержка не подходит — мягко скажи, что "
    "в базе по этому запросу ничего не нашлось. Язык ответа — русский."
)

_CATEGORY_HINT_SYSTEM = (
    "Ты классификатор внутренних HR/IT вопросов TechCorp. По вопросу сотрудника выдай ОДНУ короткую "
    "фразу на русском (до 12 слов): в какой области политики компании он скорее всего (отпуск, ДМС, "
    "оборудование, офис, зарплата и т.д.). Не перечисляй категории из списка дословно, если не уверен — "
    "опиши тему своими словами. Только фраза, без пояснений и нумерации."
)


class OpenAILLMProvider(LLMProvider):
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(api_key=settings.openai_api_key or None, base_url=settings.openai_base_url)
        self._model = settings.llm_model

    def generate_answer(self, question: str, context_hits: list[SearchHit]) -> GeneratedAnswer:
        if not context_hits:
            return GeneratedAnswer(
                text="Я не нашёл информацию по этому вопросу.",
                confidence=ConfidenceLevel.NONE,
            )

        context_blocks: list[str] = []
        for i, h in enumerate(context_hits, start=1):
            block = f"Фрагмент {i}.\nQ: {h.question}\nA: {h.answer}"
            context_blocks.append(block)
        context = "\n\n".join(context_blocks)

        n = len(context_hits)
        user_message = (
            f"Самые релевантные выдержки ({n} шт., по убыванию полезности):\n{context}\n\n"
            f"Вопрос сотрудника: {question}\n\n"
            "Ответь 4–8 предложениями, дружелюбно. Если вопрос составной — дай два логических абзаца или явные части "
            "ответа (сначала как заказать / куда писать — строго по A: с каналами и сроками как в тексте; затем что "
            "положено по оборудованию — только то, что прямо сказано в соответствующем A:, без выдуманных каталогов). "
            "Не подменяй конкретные названия из A: обобщениями. Если чего-то нет в выдержках — не придумывай, скажи "
            "что в базе об этом нет."
        )

        if self._settings.log_prompts:
            logger.info(
                "Запрос к модели: модель=%s символов_в_user=%d",
                self._model,
                len(user_message),
            )

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.28,
        )
        choice = completion.choices[0].message.content or ""
        text = choice.strip()
        if self._settings.log_prompts:
            logger.info("Ответ модели: символов=%d", len(text))

        return GeneratedAnswer(text=text, confidence=ConfidenceLevel.MEDIUM)

    def infer_category_hint(self, question: str, categories: list[str]) -> str:
        cat_block = ", ".join(categories)
        user_message = (
            f"Известные категории базы знаний (идентификаторы): {cat_block}\n\n"
            f"Вопрос сотрудника: {question}"
        )
        if self._settings.log_prompts:
            logger.info("Подсказка категории: модель=%s", self._model)

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _CATEGORY_HINT_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=80,
        )
        hint = (completion.choices[0].message.content or "").strip()
        if self._settings.log_prompts:
            logger.info("Ответ подсказки категории: символов=%d", len(hint))
        return hint or "общий вопрос сотрудника"
