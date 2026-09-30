"""src.RAG.embedding_service

Сервис создания эмбеддингов.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import get_config
from src.interfaces import EmbeddingModelProtocol

config = get_config()
embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)

# Размер пакета для генерации эмбеддингов. Кодирование больших списков
# чанков выполняется пакетами заданного размера, чтобы избежать
# избыточного потребления памяти и улучшить утилизацию модели.
EMBEDDING_BATCH_SIZE = 512


def generate_embeddings(
    model: EmbeddingModelProtocol, chunks: list[str]
) -> np.ndarray:
    """Генерирует эмбеддинги для списка текстовых чанков.

    Использует переданную модель для кодирования чанков в векторы.
    Возвращает массив NumPy, где каждая строка — эмбеддинг соответствующего
    чанка. Подходит для последующей индексации в векторной БД.

    Args:
        model: Экземпляр модели SentenceTransformer.
        chunks: Список текстовых чанков для векторизации.

    Returns:
        np.ndarray: Массив эмбеддингов формы (len(chunks), dim), где dim —
        размерность вектора модели.
    """
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings


def generate_embeddings_batch(
    model: EmbeddingModelProtocol,
    chunks: list[str],
    batch_size: int = EMBEDDING_BATCH_SIZE
) -> list[np.ndarray]:
    """Генерирует эмбеддинги для списка чанков пакетами.

    Кодирует весь список чанков, разбивая его на пакеты размером batch_size,
    и возвращает плоский список эмбеддингов (по одному вектору на чанк).
    Пакетная обработка существенно сокращает число вызовов encode
    по сравнению с поштучной векторизацией, что ускоряет индексацию
    больших корпусов.

    Args:
        model: Экземпляр модели SentenceTransformer.
        chunks: Список текстовых чанков для векторизации.
        batch_size: Размер одного пакета. По умолчанию — EMBEDDING_BATCH_SIZE.

    Returns:
        list[np.ndarray]: Список эмбеддингов, где каждый элемент — вектор
        соответствующего чанка.
    """
    if not chunks:
        return []

    embeddings: list[np.ndarray] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_embeddings = generate_embeddings(model, batch)
        embeddings.extend(
            [batch_embeddings[j] for j in range(len(batch))]
        )
    return embeddings
