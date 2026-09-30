"""src.vector_db.base"""

from abc import ABC, abstractmethod

from src.interfaces import DocumentProtocol
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
    async def add(self, data: DocumentProtocol | list[DocumentProtocol]) -> None:
        """Принимает один  объект или список структуры данных Document,
        определенной в пакете RAG, содержащей данные, которые должны быть
        добавлены в векторную БД.
        """
        pass

    @abstractmethod
    async def delete_by_source(self, source_id: str) -> None:
        """Удаляет записи коллекции, относящиеся к указанному источнику.

        Используется при инкрементальной индексации для исключения
        дубликатов при перезаписи изменённого файла-источника.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Закрывает клиент векторной БД, корректно выгружая
        все незавершённые данные (например, HNSW-индекс) на диск.

        Вызывается при завершении работы приложения (graceful shutdown).
        """
        pass
