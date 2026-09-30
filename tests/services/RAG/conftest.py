"""tests.services.RAG.conftest

Фикстуры для тестов пакета tests.services.RAG.

Содержит общие фикстуры сервисов RAG: мок конфигурации приложения,
мок модели эмбеддингов, а также временные директории и файлы
внутренней документации для изоляции тестов от реального репозитория.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import Config


@pytest.fixture
def test_doc_config(tmp_path: Path) -> Config:
    """Возвращает конфигурацию с временной директорией документов.

    Устанавливает ``DOC_DIR`` во временную папку pytest, что позволяет
    тестировать RAG‑сервисы (например, проверку типа файла) в изоляции
    от реального репозитория документации.

    Returns:
        Config: Объект конфигурации с DOC_DIR во временной директории.
    """
    return Config(DOC_DIR=tmp_path)


@pytest.fixture
def test_doc_file_path(tmp_path: Path) -> Path:
    """Возвращает путь к тестовому файлу документа внутри временной DOC_DIR.

    Сам файл на диске не создаётся: для unit‑тестов проверки типа файла
    достаточно пути, так как определение MIME‑типа мокируется.

    Returns:
        Path: Путь к тестовому файлу ``test.txt`` внутри временной папки.
    """
    return tmp_path / "test.txt"


@pytest.fixture
def test_readable_file(tmp_path: Path) -> Path:
    """Создаёт и возвращает реальный читаемый файл документа.

    Записывает во временную папку файл ``test.txt`` с несколькими абзацами,
    который можно реально прочитать в тестах doc_read_service.

    Returns:
        Path: Путь к реально существующему читаемому файлу.
    """
    file_path = tmp_path / "test.txt"
    file_path.write_text(
        "Первый абзац.\n\nВторой абзац.",
        encoding="utf-8",
    )
    return file_path


@pytest.fixture
def mock_config() -> MagicMock:
    """Возвращает мок конфигурации приложения.

    Подменяет Config, чтобы тесты были изолированы от реального
    репозитория документации и настроек эмбеддингов.

    Returns:
        MagicMock: Мок конфигурации с предопределёнными атрибутами.
    """
    config = MagicMock()
    config.EMBEDDING_MODEL = "mock-embedding-model"
    config.DOC_DIR = Path("/fake/docs")
    return config


@pytest.fixture
def mock_model() -> MagicMock:
    """Возвращает мок модели эмбеддингов.

    Мок заменяет реальную модель SentenceTransformer и возвращает
    предопределённый массив векторов, что исключает внешние зависимости.

    Returns:
        MagicMock: Мок модели с предопределённым результатом encode.
    """
    model = MagicMock()
    model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
    return model
