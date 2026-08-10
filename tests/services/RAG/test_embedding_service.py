"""tests.RAG.test_embedding_service

Тесты сервиса создания эмбеддингов для RAG‑конвейера.
Используют мок модели SentenceTransformer, чтобы избежать
загрузки реальной модели и сэкономить ресурсы.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from src.RAG.embedding_service import generate_embeddings


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
