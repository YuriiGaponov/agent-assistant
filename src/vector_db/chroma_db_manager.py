"""src.vector_db.chroma_db_manager

Реализация менеджера векторной БД для Chroma, реализующая интерфейс
BaseDBManager. Инкапсулирует логику создания коллекции, подготовки
чанков и их сохранения в ChromaDB с метаданными
(включая идентификаторы, текст и эмбеддинги).
"""

from typing import Any

import numpy as np
from chromadb import Collection

from src.interfaces import DocumentProtocol
from src.logger import logger
from src.vector_db.base import BaseDBManager

# Размер порции документов при записи в векторную БД. Разбиение большого
# списка на порции снижает пик потребления памяти и упрощает диагностику
# ошибок при массовой записи.
UPSERT_BATCH_SIZE = 2000


class ChromaDBManager(BaseDBManager):
    """Реализация менеджера векторной БД для Chroma."""

    def __init__(self, config):
        super().__init__(config)
        self._collection = None
        self._closed = False
        logger.debug(f"Инициализирован {ChromaDBManager}")

    @property
    async def collection(self) -> Collection:
        """Ленивая инициализация коллекции: создаётся при первом обращении."""
        if self._collection is None:
            self._collection = await self.create_collection()
        return self._collection

    def _normalize_embeddings(self, emb_array: np.ndarray):
        """Приводит эмбеддинг к формату, ожидаемому Chroma.

        Преобразует вектор эмбеддинга из numpy-массива в список
        Python-чисел (list[float]). Корректно обрабатывает как
        одномерный вектор формы (dim,), так и двумерный массив
        формы (1, dim), содержащий один вектор.

        Args:
            emb_array: Входной эмбеддинг (np.ndarray). Может быть
                одномерным вектором (dim,) либо двумерным массивом
                (1, dim) с одним вектором.

        Returns:
            list[float]: Вектор эмбеддинга в виде списка чисел с плавающей
            точкой.
        """
        if emb_array.ndim == 1:
            return emb_array.tolist()
        # Формат (1, dim): извлекаем единственный вектор.
        return emb_array[0].tolist()

    async def create_collection(self, name='Agent-Assistant') -> Collection:
        """Создаёт или получает коллекцию в Chroma.

        Args:
            name: Имя коллекции. Если не указано, используется значение
            по умолчанию.

        Returns:
            Collection: Объект коллекции Chroma.
        """
        # Явно задаём метрику HNSW через метаданные. Параметр metadata
        # применяется только при создании новой коллекции; для уже
        # существующей коллекции Chroma его игнорирует. Это делает
        # конфигурацию индекса детерминированной между записью и чтением.
        collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        return collection

    async def add(self, data: DocumentProtocol | list[DocumentProtocol]):
        """Добавляет документы (с чанками и эмбеддингами) в векторную БД.
        Перезаписывает данные с одинаковым id.

        Для каждого документа:
          - проходит по чанкам и соответствующим эмбеддингам;
          - формирует уникальные ID;
          - передаёт данные в Chroma через collection.add.

        Запись выполняется порциями размером UPSERT_BATCH_SIZE, чтобы
        снизить пик потребления памяти при массовой индексации.

        Args:
            data: Один объект Document или список объектов.
        """
        docs = data if isinstance(data, list) else [data]
        collection = await self.collection

        # Разбиваем документы на порции и выполняем upsert для каждой.
        for i in range(0, len(docs), UPSERT_BATCH_SIZE):
            batch = docs[i:i + UPSERT_BATCH_SIZE]
            ids: list[str] = []
            documents: list[str] = []
            embeddings: list[np.ndarray] = []
            metadata: list[dict[str, Any]] = []

            for doc in batch:
                ids.append(f'id_{doc.id}')
                documents.append(doc.chunk)
                embeddings.append(self._normalize_embeddings(doc.embedding))
                meta: dict[str, Any] = {
                    'embedding_model': doc.meta.embedding_model,
                    'source_id': doc.meta.source.id,
                    "source_path": str(doc.meta.source.path),
                    "source_type": doc.meta.source.type,
                    "content_hash": doc.meta.source.content_hash,
                    "updated_at": doc.meta.source.updated_at.isoformat()
                }
                metadata.append(meta)

            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadata
            )

        logger.debug('документы добавлены')

    async def delete_by_source(self, source_id: str):
        """Удаляет записи коллекции, относящиеся к указанному источнику.

        Используется при инкрементальной индексации: перед перезаписью
        изменённого файла-источника удаляются его старые записи,
        чтобы исключить дубликаты.

        Args:
            source_id: Идентификатор файла-источника, чьи записи удаляются.
        """
        collection = await self.collection
        collection.delete(where={'source_id': source_id})
        logger.debug('записи источника удалены', source_id=source_id)

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
                (чем меньше — тем ближе вектор).
        """
        collection = await self.collection
        # Приводим вектор запроса к единому формату list[float] —
        # аналогично тому, как это делается при записи векторов.
        normalized = self._normalize_embeddings(embeddings)
        results = collection.query(query_embeddings=[normalized])
        return results

    async def close(self) -> None:
        """Закрывает клиент векторной БД, корректно выгружая
        незавершённые данные (например, HNSW-индекс) на диск.

        Защищает от повторного вызова: повторное закрытие уже
        освобождённого клиента игнорируется.
        """
        if self._closed:
            logger.debug('клиент векторной БД уже закрыт, пропуск')
            return
        self._closed = True
        try:
            self.client.close()
            logger.debug('клиент векторной БД закрыт')
        except Exception as e:
            logger.error(f'ошибка при закрытии клиента векторной БД: {e}')
            raise
