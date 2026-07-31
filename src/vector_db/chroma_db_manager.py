"""src.vector_db.chroma_db_manager

Реализация менеджера векторной БД для Chroma, реализующая интерфейс
BaseDBManager. Инкапсулирует логику создания коллекции, подготовки
чанков и их сохранения в ChromaDB с метаданными
(включая идентификаторы, текст и эмбеддинги).
"""

import numpy as np
from chromadb import Collection

from src.interfaces import Document
from src.logger import logger
from src.vector_db.base import BaseDBManager


class ChromaDBManager(BaseDBManager):
    """Реализация менеджера векторной БД для Chroma."""

    def __init__(self, config):
        super().__init__(config)
        self._collection = None

    @property
    async def collection(self) -> Collection:
        """Ленивая инициализация коллекции: создаётся при первом обращении."""
        if self._collection is None:
            self._collection = await self.create_collection()
        return self._collection

    async def create_collection(self, name='Agent-Assistant') -> Collection:
        """Создаёт или получает коллекцию в Chroma.

        Args:
            name: Имя коллекции. Если не указано, используется значение
            по умолчанию.

        Returns:
            Collection: Объект коллекции Chroma.
        """
        collection = self.client.get_or_create_collection(name)
        return collection

    async def add(self, data: Document | list[Document]):
        """Добавляет документы (с чанками и эмбеддингами) в векторную БД.

        Для каждого документа:
          - проходит по чанкам и соответствующим эмбеддингам;
          - формирует уникальные ID;
          - передаёт данные в Chroma через collection.add.

        Args:
            data: Один объект Document или список объектов.
        """
        docs = data if isinstance(data, list) else [data]
        collection = await self.collection
        ids: list[str] = []
        documents: list[str] = []
        embeddings: list[np.ndarray] = []
        id = 0

        for doc in docs:
            if doc.embeddings is not None:
                # добавить лог для пустого embeddings
                for (chunk, emb) in zip(doc.chunks, doc.embeddings):
                    id += 1
                    ids.append(f'id_{id}')
                    documents.append(chunk)
                    embeddings.append(emb.tolist())

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents
        )

        logger.debug('документы добавлены')

    async def search(self, embeddings: np.ndarray):
        """Выполняет семантический поиск ближайших по вектору.

        Args:
            query_vector: Вектор запроса (np.ndarray).

        Returns:
            Словарь с результатами поиска в формате, возвращаемом Chroma:
            {
                "ids": [[str, ...]],
                "documents": [[str, ...]],
                "metadatas": [[dict | None, ...] | None],
                "distances": [[float, ...]],
                ...
            }

            Основные поля:
              - ids: вложенные списки ID найденных записей;
              - documents: вложенные списки текстов документов;
              - metadatas: вложенные списки метаданных (может быть None);
              - distances: вложенные списки расстояний
                (чем меньше — тем ближе вектор).
        """
        collection = await self.collection
        results = collection.query(embeddings)
        return results
