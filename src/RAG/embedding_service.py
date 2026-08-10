"""src.RAG.embedding_service

Сервис создания эмбеддингов.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import get_config
from src.interfaces import EmbeddingModelProtocol

config = get_config()
embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)


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
