"""src.RAG.search_service

Модуль семантического поиска для RAG‑конвейера Agent‑Assistant.
Предоставляет сервис для выполнения семантического поиска
по векторной БД на основе текстового запроса пользователя.
"""


from src.interfaces import (
    VectorDBManagerInterface, EmbeddingModelProtocol
)
from src.RAG.embedding_service import generate_embeddings


async def semantic_search(
    search_request: str,
    embedding_model: EmbeddingModelProtocol,
    vectorDB: VectorDBManagerInterface
) -> dict:
    """Выполняет семантический поиск по векторной базе данных.

    Оркестрирует полный цикл поиска:
      1. Получает модель эмбеддингов и менеджер векторной БД из состояния
         приложения.
      2. Генерирует эмбеддинг для запроса пользователя.
      3. Выполняет поиск в векторной БД.
      4. Возвращает список релевантных чанков с метаданными.

    Args:
        search_request: Объект запроса, содержащий текст запроса и опциональные
                        параметры поиска (top_k, фильтры и т.п.). Конкретные
                        параметры используются внутри vectorDB.search.
        embedding_model: Реализация модели эмбеддингов, соответствующая
                         интерфейсу EmbeddingModelProtocol.
        vectorDB: Менеджер векторной БД, соответствующий интерфейсу
                  VectorDBManagerInterface.

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
    embeddings = generate_embeddings(embedding_model, [search_request])
    results = await vectorDB.search(embeddings)

    return results
