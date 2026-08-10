"""src.agent.base"""

from abc import ABC, abstractmethod

from src.logger import logger
from src.config import Config
from src.interfaces import UserQueryProtocol
from src.agent.state import State


class BaseAgent(ABC):
    """Базовый класс менеджера LLM, содержащий обязательные
    для конкретных реализаций атрибуты и методы"""

    def __init__(self, config: Config):
        self.model = config.MAIN_LLM

    async def init_state(self, query: UserQueryProtocol) -> State:
        """Инициализация начального состояния.

        Выполнена асинхронная реализация, т.к. в следующих версиях
        запланировано получение сохраненного состояния из реляционной БД.
        """
        state = State(
            query=query.query
        )
        logger.info(f'получено начальное состояние: {state}')

        return state

    @abstractmethod
    async def health_check(self) -> bool:
        """Возвращает True, если LLM доступна и отвечает за разумное время."""
        pass

    @abstractmethod
    async def chat(self, query: UserQueryProtocol):
        """Взаимодействие с LLM в виде диалога. Принимает список сообщений,
        возвращает обновленный список сообщений.
        """
        pass
