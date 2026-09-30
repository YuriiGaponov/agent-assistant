"""src.vector_db.client

Модуль инициализации клиента векторной БД.
Предоставляет фабрику для создания экземпляра клиента в зависимости
от конфигурации проекта. На текущем этапе поддерживает только ChromaDB;
в будущем может быть расширен поддержкой других векторных СУБД.
"""

import chromadb
from chromadb.api import ClientAPI

from src.config import Config, VectorDB


def init_client(config: Config) -> ClientAPI:
    """Инициализирует клиент векторной БД согласно конфигурации.

    Args:
        config: Объект конфигурации с указанием типа векторной БД
            (config.VECTOR_DB).

    Returns:
        ClientAPI: Экземпляр клиента векторной БД (на текущий момент — Chroma).

    Raises:
        ValueError: Если указан неподдерживаемый тип векторной БД.
    """
    if config.VECTOR_DB == VectorDB.CHROMA:
        return chromadb.PersistentClient(
            path=config.VECTOR_DB_PATH
        )

    raise ValueError(f"Неподдерживаемая векторная БД: {config.VECTOR_DB}")
