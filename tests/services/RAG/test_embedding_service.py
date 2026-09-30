"""tests.RAG.test_embedding_service

Тесты сервиса создания эмбеддингов для RAG‑конвейера.
Используют мок модели SentenceTransformer, чтобы избежать
загрузки реальной модели и сэкономить ресурсы.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.RAG.embedding_service import (
    generate_embeddings,
    generate_embeddings_batch,
    EMBEDDING_BATCH_SIZE,
)


def test_generate_embeddings_returns_expected_shape() -> None:
    """Проверка, что generate_embeddings возвращает массив правильной формы.

    Мокируем модель, чтобы она возвращала фиксированный массив
    заданной размерности.
    """
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    chunks = ["Первый чанк.", "Второй чанк."]
    result = generate_embeddings(mock_model, chunks)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)
    mock_model.encode.assert_called_once_with(chunks, convert_to_numpy=True)


def test_generate_embeddings_single_chunk() -> None:
    """Один чанк возвращает массив формы (1, dim)."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.5, 0.5]])

    result = generate_embeddings(mock_model, ["Один чанк."])

    assert result.shape == (1, 2)


def test_generate_embeddings_empty_chunks() -> None:
    """Пустой список чанков — модель вызывается с пустым списком."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.empty((0, 384))

    result = generate_embeddings(mock_model, [])

    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 384)
    mock_model.encode.assert_called_once_with([], convert_to_numpy=True)


def test_generate_embeddings_calls_model_encode() -> None:
    """Проверка, что модель вызывается ровно один раз с переданными чанками."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1], [0.2], [0.3]])

    chunks = ["A", "B", "C"]
    generate_embeddings(mock_model, chunks)

    mock_model.encode.assert_called_once_with(chunks, convert_to_numpy=True)


def test_generate_embeddings_uses_convert_to_numpy() -> None:
    """Проверка, что параметр convert_to_numpy=True передаётся в модель."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[1.0]])

    generate_embeddings(mock_model, ["Тест"])

    mock_model.encode.assert_called_once_with(["Тест"], convert_to_numpy=True)


def test_generate_embeddings_different_dimensions() -> None:
    """Проверка работы с разной размерностью эмбеддингов."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]])

    result = generate_embeddings(mock_model, ["Чанк"])

    assert result.shape[1] == 4


# ─── Тесты generate_embeddings_batch ─────────────────────────────────────


def test_generate_embeddings_batch_returns_list() -> None:
    """generate_embeddings_batch возвращает список векторов по числу чанков."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

    chunks = ["Чанк 1.", "Чанк 2."]
    result = generate_embeddings_batch(mock_model, chunks)

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(emb, np.ndarray) for emb in result)


def test_generate_embeddings_batch_splits_by_batch_size() -> None:
    """Чанки разбиваются на пакеты заданного размера."""
    mock_model = MagicMock()
    mock_model.encode.side_effect = [
        np.array([[0.1, 0.2]]),
        np.array([[0.3, 0.4]]),
        np.array([[0.5, 0.6]]),
    ]

    chunks = ["Чанк 1.", "Чанк 2.", "Чанк 3."]
    result = generate_embeddings_batch(mock_model, chunks, batch_size=1)

    assert len(result) == 3
    # Три вызова encode: по одному пакету на каждый чанк.
    assert mock_model.encode.call_count == 3


def test_generate_embeddings_batch_default_batch_size_constant() -> None:
    """Размер пакета по умолчанию равен EMBEDDING_BATCH_SIZE."""
    assert EMBEDDING_BATCH_SIZE == 512


def test_generate_embeddings_batch_empty_chunks() -> None:
    """Пустой список чанков возвращает пустой список без вызова encode."""
    mock_model = MagicMock()

    result = generate_embeddings_batch(mock_model, [])

    assert result == []
    mock_model.encode.assert_not_called()


def test_generate_embeddings_batch_preserves_order() -> None:
    """Эмбеддинги сохраняют порядок исходных чанков."""
    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

    chunks = ["Первый.", "Второй."]
    result = generate_embeddings_batch(mock_model, chunks)

    assert np.array_equal(result[0], np.array([0.1, 0.2]))
    assert np.array_equal(result[1], np.array([0.3, 0.4]))
