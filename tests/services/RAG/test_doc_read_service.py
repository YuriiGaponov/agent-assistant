"""tests.RAG.test_doc_read_service

Тесты для read_doc и read_all_docs — основных сценариев чтения
документов из директории внутренней документации Agent‑Assistant.

Проверяются:
- Успешное чтение одного файла с генерацией эмбеддингов
- Успешное чтение нескольких файлов
- Обработка пустой директории
- Ошибка при недоступной директории
- Проброс ошибки при чтении одного из файлов
- Полностью изолированные тесты с моками
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.config import Config
from src.RAG.doc_read_service import read_doc, read_all_docs
from src.RAG.schemas import Document


# ─── Тесты read_doc ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_doc_success(tmp_path: Path) -> None:
    """Успешное чтение файла с несколькими абзацами и эмбеддингами.

    Создаёт временный файл, мокирует модель SentenceTransformer
    и проверяет, что read_doc возвращает Document с корректными
    чанками и эмбеддингами.
    """
    file_path = tmp_path / "test.txt"
    file_path.write_text(
        "Первый абзац.\n\nВторой абзац.\n\nТретий абзац.",
        encoding="utf-8",
    )

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
    ])

    doc = await read_doc(file_path, mock_model)

    assert isinstance(doc, Document)
    assert len(doc.chunks) == 3
    assert doc.chunks[0] == "Первый абзац."
    assert doc.chunks[1] == "Второй абзац."
    assert doc.chunks[2] == "Третий абзац."
    assert doc.embeddings is not None
    assert doc.embeddings.shape == (3, 2)


@pytest.mark.asyncio
async def test_read_doc_empty_file(tmp_path: Path) -> None:
    """Чтение пустого файла возвращает Document с пустыми чанками и None эмбеддингами."""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    mock_model = MagicMock()
    mock_model.encode.return_value = np.empty((0, 384))

    doc = await read_doc(file_path, mock_model)

    assert isinstance(doc, Document)
    assert doc.chunks == []
    assert doc.embeddings is not None
    assert doc.embeddings.shape == (0, 384)


@pytest.mark.asyncio
async def test_read_doc_single_paragraph(tmp_path: Path) -> None:
    """Чтение файла с единственным абзацем возвращает один чанк."""
    file_path = tmp_path / "single.txt"
    file_path.write_text("Единственный абзац.", encoding="utf-8")

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.42, 0.42]])

    doc = await read_doc(file_path, mock_model)

    assert isinstance(doc, Document)
    assert len(doc.chunks) == 1
    assert doc.chunks[0] == "Единственный абзац."
    assert doc.embeddings.shape == (1, 2)


@pytest.mark.asyncio
async def test_read_doc_file_not_found() -> None:
    """Несуществующий файл пробрасывает FileNotFoundError."""
    non_existent = Path("/non/existent/doc.txt")
    mock_model = MagicMock()

    with pytest.raises(FileNotFoundError):
        await read_doc(non_existent, mock_model)


@pytest.mark.asyncio
async def test_read_doc_calls_generate_embeddings(tmp_path: Path) -> None:
    """Проверка, что read_doc вызывает generate_embeddings с правильными аргументами."""
    file_path = tmp_path / "embed_test.txt"
    file_path.write_text("Чанк A.\n\nЧанк B.", encoding="utf-8")

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

    doc = await read_doc(file_path, mock_model)

    mock_model.encode.assert_called_once_with(
        ["Чанк A.", "Чанк B."], convert_to_numpy=True
    )
    assert doc.embeddings.shape == (2, 2)


# ─── Тесты read_all_docs ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_all_docs_success(tmp_path: Path) -> None:
    """Успешное чтение всех документов из директории."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    file1 = doc_dir / "doc1.txt"
    file2 = doc_dir / "doc2.txt"
    file1.write_text("Первый абзац.\n\nВторой абзац.", encoding="utf-8")
    file2.write_text("Третий абзац.", encoding="utf-8")

    config = Config(DOC_DIR=doc_dir)
    mock_model = MagicMock()

    def encode_side_effect(chunks, **kwargs):
        if chunks == ["Первый абзац.", "Второй абзац."]:
            return np.array([[0.1, 0.2], [0.3, 0.4]])
        elif chunks == ["Третий абзац."]:
            return np.array([[0.5, 0.6]])
        return np.empty((0, 2))

    mock_model.encode.side_effect = encode_side_effect

    docs = await read_all_docs(config, mock_model)

    assert len(docs) == 2
    assert all(isinstance(doc, Document) for doc in docs)
    assert docs[0].chunks == ["Первый абзац.", "Второй абзац."]
    assert docs[1].chunks == ["Третий абзац."]
    assert docs[0].embeddings.shape == (2, 2)
    assert docs[1].embeddings.shape == (1, 2)


@pytest.mark.asyncio
async def test_read_all_docs_empty_directory(tmp_path: Path) -> None:
    """Пустая директория возвращает пустой список документов."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()

    config = Config(DOC_DIR=doc_dir)
    mock_model = MagicMock()

    docs = await read_all_docs(config, mock_model)

    assert docs == []


@pytest.mark.asyncio
async def test_read_all_docs_raises_when_directory_missing(
    tmp_path: Path,
) -> None:
    """Несуществующая директория вызывает NotADirectoryError."""
    nonexistent = tmp_path / "does_not_exist"
    config = Config(DOC_DIR=nonexistent)
    mock_model = MagicMock()

    msg = "Папка внутренней документации недоступна"
    with pytest.raises(NotADirectoryError, match=msg):
        await read_all_docs(config, mock_model)


@pytest.mark.asyncio
async def test_read_all_docs_raises_when_path_is_file(
    tmp_path: Path,
) -> None:
    """Путь к файлу вместо директории вызывает NotADirectoryError."""
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("dummy", encoding="utf-8")
    config = Config(DOC_DIR=file_path)
    mock_model = MagicMock()

    msg = "Папка внутренней документации недоступна"
    with pytest.raises(NotADirectoryError, match=msg):
        await read_all_docs(config, mock_model)


@pytest.mark.asyncio
async def test_read_all_docs_propagates_file_read_error(
    tmp_path: Path,
) -> None:
    """Ошибка чтения одного из файлов пробрасывается наверх."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "good.txt").write_text("Хороший файл.", encoding="utf-8")
    (doc_dir / "bad.txt").write_text("Плохой файл.", encoding="utf-8")

    config = Config(DOC_DIR=doc_dir)
    mock_model = MagicMock()

    read_doc_path = "src.RAG.doc_read_service.read_doc"

    async def mock_read_doc(path: Path, model) -> Document:
        if "bad" in path.name:
            raise RuntimeError(f"Ошибка чтения: {path.name}")
        return Document(chunks=["Хороший файл."])

    with patch(read_doc_path, new=mock_read_doc):
        with pytest.raises(RuntimeError, match="Ошибка чтения: bad.txt"):
            await read_all_docs(config, mock_model)


# ─── Mock-based тесты (полная изоляция) ─────────────────────────────────


@pytest.mark.asyncio
async def test_read_all_docs_with_mocks() -> None:
    """Полностью изолированный тест с моками всех зависимостей.

    Проверяет, что read_all_docs корректно вызывает doc_dir_exist,
    get_doc_files и read_doc в правильной последовательности
    и возвращает результат, собранный через asyncio.gather.
    """
    config = Config(DOC_DIR=Path("/fake/docs"))
    mock_model = MagicMock()
    fake_paths = [Path("/fake/docs/a.txt"), Path("/fake/docs/b.txt")]
    fake_docs = [
        Document(chunks=["A"], embeddings=np.array([[0.1]])),
        Document(chunks=["B"], embeddings=np.array([[0.2]])),
    ]

    with (
        patch(
            "src.RAG.doc_read_service.doc_dir_exist",
            return_value=True,
        ) as mock_exist,
        patch(
            "src.RAG.doc_read_service.get_doc_files",
            return_value=fake_paths,
        ) as mock_get_files,
        patch(
            "src.RAG.doc_read_service.read_doc",
            new_callable=AsyncMock,
        ) as mock_read_doc,
    ):
        mock_read_doc.side_effect = fake_docs

        result = await read_all_docs(config, mock_model)

        assert result == fake_docs
        mock_exist.assert_called_once_with(config)
        mock_get_files.assert_called_once_with(config)
        assert mock_read_doc.call_count == 2
        mock_read_doc.assert_any_call(fake_paths[0], mock_model)
        mock_read_doc.assert_any_call(fake_paths[1], mock_model)
