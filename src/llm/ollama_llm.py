"""src.llm.ollama_llm

Реализация менеджера LLM для работы с Ollama в составе Agent‑Assistant.
Предоставляет асинхронный интерфейс взаимодействия с локальной LLM через
Ollama: создание клиента модели, привязку инструментов, вызов модели
и проверку доступности.
"""

import asyncio

import ollama
from langchain_ollama import ChatOllama

from src.logger import logger
from src.llm.base import BaseLLMManager


class OllamaLLMManager(BaseLLMManager):
    """Реализация менеджера LLM для работы с Ollama."""

    def _create_model(self):
        """Создаёт клиент ChatOllama для выбранной модели.

        Returns:
            ChatOllama: Клиент модели LangChain для Ollama.
        """
        return ChatOllama(model=self.model.value)

    async def health_check(self) -> bool:
        """Проверяет доступность модели в Ollama.

        Выполняет тестовый запрос «2+2=?» и убеждается,
        что получен непустой ответ.
        Запрос выполняется в отдельном потоке, чтобы не блокировать event loop.

        Returns:
            bool: True, если модель доступна и возвращает осмысленный ответ;
                  иначе False.
        """
        loop = asyncio.get_running_loop()
        try:
            def sync_health_check() -> bool:
                try:
                    response = ollama.generate(
                        model=self.model.value,
                        prompt="2+2=?",
                    )
                    text = response.get("response")
                    return bool(text and len(text.strip()) > 0)
                except Exception as e:
                    logger.error(f'Ошибка при проверке доступности LLM: {e}')
                    return False

            return await loop.run_in_executor(None, sync_health_check)
        except Exception as e:
            logger.error(f'Ошибка при проверке доступности LLM: {e}')
            return False
