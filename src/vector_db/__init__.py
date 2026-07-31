"""src.vector_db.__init__.py

Пакет реализации взаимодействия с векторной базой данных для Agent‑Assistant.
Предоставляет унифицированный интерфейс для работы с векторным хранилищем:
создание коллекции, добавление векторов, поиск по схожести и управление
метаданными.

Модуль выступает в роли фабрики: по конфигурации возвращает реализацию
менеджера векторной БД (например, ChromaDBManager), которая реализует
общий интерфейс BaseDBManager. Это позволяет подменять бэкенд без изменения
кода в RAG‑сервисах.
"""

from src.logger import logger
from src.config import Config, VectorDB
from src.vector_db.base import BaseDBManager
from src.vector_db.chroma_db_manager import ChromaDBManager


def get_vectorDB_manager(config: Config) -> BaseDBManager:
    """Возвращает экземпляр менеджера векторной БД согласно конфигурации.

    Фабрика выбирает реализацию в зависимости от значения config.VECTOR_DB.
    На текущий момент поддерживается только Chroma; при добавлении новых
    бэкендов достаточно добавить новые ветки if/elif.

    Args:
        config: Объект конфигурации проекта с указанием типа векторной БД.

    Returns:
        BaseDBManager: Реализация менеджера векторной БД, готовая к работе.

    Raises:
        ValueError: Если указан неподдерживаемый тип векторной БД.
    """
    if config.VECTOR_DB == VectorDB.CHROMA:
        logger.debug("Инициализирован ChromaDBManager")
        return ChromaDBManager(config)

    raise ValueError(f"Неподдерживаемая векторная БД: {config.VECTOR_DB}")
