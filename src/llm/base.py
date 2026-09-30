"""src.llm.base

Базовый класс менеджера LLM для проекта Agent‑Assistant.

Инкапсулирует низкоуровневую логику взаимодействия с моделью:
создание клиента модели, привязку инструментов, вызов модели
и проверку доступности. Реализации для конкретных провайдеров
(Ollama, Mistral и т.д.) наследуют этот класс и задают способ
создания клиента в методе _create_model.
"""

from abc import ABC, abstractmethod

from langchain_core.tools import BaseTool

from src.config import Config


class BaseLLMManager(ABC):
    """Базовый класс менеджера LLM.

    Содержит общую логику управления моделью:
    - хранение выбранной модели (config.MAIN_LLM);
    - ленивое создание клиента модели через _create_model;
    - привязку инструментов через bind_tools;
    - вызов модели через ainvoke;
    - абстрактный health_check для проверки доступности.

    Args:
        config: Конфигурация проекта (Config).
    """

    def __init__(self, config: Config):
        if config.MAIN_LLM is None:
            raise ValueError("Не установлен источник LLM")
        self.config = config
        self.model = config.MAIN_LLM
        self._model = None
        self._tools: list[BaseTool] = []

    @property
    def llm(self):
        """Клиент модели LangChain (лениво создаётся и кэшируется).

        Returns:
            Клиент модели с привязанными инструментами (если bind_tools
            был вызван) либо сырой клиент модели.
        """
        if self._model is None:
            self._model = self._create_model()
        return self._model

    @abstractmethod
    def _create_model(self):
        """Создаёт клиент модели для конкретного провайдера.

        Returns:
            Клиент модели LangChain.
        """
        pass

    def bind_tools(self, tools: list):
        """Привязывает инструменты к модели.

        Пересоздаёт клиент модели с привязанными инструментами,
        чтобы модель могла самостоятельно принимать решение об их вызове.

        Args:
            tools: Список инструментов LangChain.

        Returns:
            BaseLLMManager: Текущий менеджер с привязанными инструментами.
        """
        self._tools = tools
        self._model = self._create_model().bind_tools(tools)
        return self

    async def ainvoke(self, messages: list):
        """Вызывает модель с переданными сообщениями.

        Args:
            messages: Список сообщений LangChain.

        Returns:
            Ответ модели (AIMessage).
        """
        return await self.llm.ainvoke(messages)

    @abstractmethod
    async def health_check(self) -> bool:
        """Проверяет доступность модели.

        Returns:
            bool: True, если модель доступна и отвечает; иначе False.
        """
        pass
