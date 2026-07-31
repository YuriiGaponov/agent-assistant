"""src.interfaces

Абстрактные классы и протоколы, определяющие контрактыы
различных модулей и пакетов, а также внешних систем проекта Agent‑Assistant.
"""

from typing import Protocol, Sequence

import numpy as np


class Document(Protocol):
    """Структура данных прочитанного документа внутренней документации,
    содержащая поля, которые необходимо внести в векторную БД.
    """
    chunks: list[str]
    embeddings: np.ndarray | None = None


class VectorSearchQueryProtocol(Protocol):
    """Протокол объекта запроса для семантического поиска в векторной БД"""
    query: str


class VectorSearchResultProtocol(Protocol):
    """Протокол объекта результата семантического поиска в векторной БД,
    возвращаемого менеджером векторной БД.
    """
    documents: Sequence[str]


class VectorDBManagerInterface(Protocol):
    """Протокол менеджера векторной БД: определяет ожидаемый интерфейс
    для добавления документов в векторное хранилище.
    """
    async def create_collection(self, name: str = 'Agent-Assistant'):
        """Получает существующую или создает новую коллекцию векторной БД
        с заданным именем name.
        """
        pass

    async def add(self, data: Document | list[Document]) -> None:
        """Принимает один  объект или список структуры данных Document,
        определенной в пакете RAG, содержащей данные, которые должны быть
        добавлены в векторную БД.
        """
        pass

    async def search(self, embeddings: np.ndarray):
        """Принимает преобразованный в эмбеддинги поисковый запрос
        для семантического поиска в векторной БД.
        """
        pass
