"""src.agent.ollama_llm

Реализация LLM‑менеджера для работы с Ollama в составе Agent‑Assistant.
Предоставляет асинхронный интерфейс взаимодействия с локальной LLM через
Ollama.
"""

import asyncio
import ollama

from src.config import Config, LLM
from src.logger import logger
from src.interfaces import (
    EmbeddingModelProtocol, UserQueryProtocol, VectorDBManagerInterface
)
from src.agent.base import BaseAgent
from src.agent.graph import build_graph


class OllamaAgent(BaseAgent):
    """Реализация LLM-менеджера для работы с Ollama"""

    def __init__(
        self,
        config: Config,
        embedding_model: EmbeddingModelProtocol,
        vector_db: VectorDBManagerInterface,
    ):
        if config.MAIN_LLM is None:
            raise ValueError("Не установлен источник LLM")
        self.model: LLM = config.MAIN_LLM
        self.graph = build_graph(config, embedding_model, vector_db)
        logger.info(
            f"OllamaLLMManager инициализирован для модели: {self.model}"
        )

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
                        model=self.model,
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

    async def chat(self, query: UserQueryProtocol) -> str:
        """Обрабатывает пользовательский запрос через граф RAG‑цикла.

        Инициализирует начальное состояние, запускает граф агента
        (цикл «вызов модели → при необходимости вызов инструментов»)
        и возвращает итоговый обобщённый ответ.

        Args:
            query: Объект запроса пользователя (UserQueryProtocol).

        Returns:
            str: Обобщённый ответ агента.
        """
        initial_state = await self.init_state(query)
        result = await self.graph.ainvoke(initial_state)
        return result.get('answer', '')
