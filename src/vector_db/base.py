"""src.vector_db.base"""

from abc import ABC, abstractmethod

from src.interfaces import Document
from src.config import Config
from src.vector_db.client import init_client


class BaseDBManager(ABC):
    """Базовый класс менеджера векторной БД, содержащий обязательные
    для конкретных реализаций атрибуты и методы"""

    def __init__(self, config: Config):
        self.client = init_client(config)

    @abstractmethod
    async def create_collection(self, name: str = 'Agent‑Assistant'):
        """Получает существующую или создает новую коллекцию векторной БД
        с заданным именем name.
        """
        pass

    @abstractmethod
    async def add(self, data: Document | list[Document]) -> None:
        """Принимает один  объект или список структуры данных Document,
        определенной в пакете RAG, содержащей данные, которые должны быть
        добавлены в векторную БД.
        """
        pass
