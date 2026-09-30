"""tests.services.RAG.test_doc_read_service

Тесты сервиса чтения внутренней документации Agent‑Assistant.

Проверяются:
- get_source: создание структуры данных файла-источника
- get_doc: создание Document на основе чанка и источника
- read_file: чтение мок-файла с генерацией списка Document
- read_all_docs: обработка директории с документами,
  включая ошибки недоступности папки и чтения файлов
"""

from collections.abc import Sequence
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.RAG.doc_read_service import (
    get_doc,
    get_source,
    read_all_docs,
    read_changed_docs,
    read_file,
    attach_embeddings,
    embed_documents,
    _deduplicate_docs,
)
from src.RAG.schemas import Document, Source


# ─── Тесты get_source ───────────────────────────────────────────────────


def test_get_source_returns_source(
    test_readable_file: Path,
    mock_config: MagicMock,
) -> None:
    """get_source возвращает объект src.RAG.schemas.Source."""
    file_type = "text/plain"
    content = test_readable_file.read_text(encoding="utf-8")

    source = get_source(test_readable_file, file_type, content)

    assert isinstance(source, Source)
    assert source.path == test_readable_file
    assert source.type == file_type


# ─── Тесты get_doc ──────────────────────────────────────────────────────


def test_get_doc_returns_document(
    test_readable_file: Path,
    mock_config: MagicMock,
) -> None:
    """get_doc возвращает объект src.RAG.schemas.Document."""
    source = get_source(
        test_readable_file,
        "text/plain",
        test_readable_file.read_text(encoding="utf-8"),
    )

    doc = get_doc("Тестовый чанк", source, mock_config)

    assert isinstance(doc, Document)
    assert doc.chunk == "Тестовый чанк"
    assert doc.meta.source == source
    assert doc.embedding is None


# ─── Тесты read_file ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_file_returns_documents(
    test_readable_file: Path,
    mock_config: MagicMock,
) -> None:
    """read_file читает мок-файл и возвращает список Document."""
    with (
        patch(
            "src.RAG.doc_read_service.get_file_type_async",
            new_callable=AsyncMock,
            return_value="text/plain",
        ),
        patch(
            "src.RAG.doc_read_service.get_file_content",
            new_callable=AsyncMock,
            return_value="Чанк A.\n\nЧанк B.",
        ),
    ):
        docs = await read_file(test_readable_file, mock_config)

    assert isinstance(docs, list)
    assert len(docs) > 0
    assert all(isinstance(doc, Document) for doc in docs)


# ─── Тесты _deduplicate_docs ────────────────────────────────────────────


def test_deduplicate_docs_removes_duplicates(
    test_readable_file: Path,
    mock_config: MagicMock,
) -> None:
    """_deduplicate_docs удаляет дубликаты чанков по chunk_hash."""
    source = get_source(
        test_readable_file,
        "text/plain",
        test_readable_file.read_text(encoding="utf-8"),
    )
    doc1 = get_doc("Одинаковый чанк", source, mock_config)
    doc2 = get_doc("Одинаковый чанк", source, mock_config)
    doc3 = get_doc("Другой чанк", source, mock_config)

    unique = _deduplicate_docs([doc1, doc2, doc3])

    assert len(unique) == 2


# ─── Тесты attach_embeddings ────────────────────────────────────────────


def test_attach_embeddings_fills_docs(
    test_readable_file: Path,
    mock_config: MagicMock,
) -> None:
    """attach_embeddings заполняет embedding у документов."""
    source = get_source(
        test_readable_file,
        "text/plain",
        test_readable_file.read_text(encoding="utf-8"),
    )
    docs = [
        get_doc("Чанк 1", source, mock_config),
        get_doc("Чанк 2", source, mock_config),
    ]
    embeddings = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]

    result = attach_embeddings(docs, embeddings)

    assert len(result) == 2
    assert np.array_equal(result[0].embedding, np.array([0.1, 0.2]))
    assert np.array_equal(result[1].embedding, np.array([0.3, 0.4]))


def test_attach_embeddings_raises_on_length_mismatch(
    test_readable_file: Path,
    mock_config: MagicMock,
) -> None:
    """attach_embeddings выбрасывает ValueError при несовпадении длин."""
    source = get_source(
        test_readable_file,
        "text/plain",
        test_readable_file.read_text(encoding="utf-8"),
    )
    docs = [get_doc("Чанк 1", source, mock_config)]
    embeddings = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]

    with pytest.raises(ValueError, match="не совпадает"):
        attach_embeddings(docs, embeddings)


# ─── Тесты embed_documents ──────────────────────────────────────────────


def test_embed_documents_uses_batch_and_attaches(
    test_readable_file: Path,
    mock_config: MagicMock,
    mock_model: MagicMock,
) -> None:
    """embed_documents дедуплицирует и заполняет эмбеддинги."""
    source = get_source(
        test_readable_file,
        "text/plain",
        test_readable_file.read_text(encoding="utf-8"),
    )
    docs = [
        get_doc("Чанк A", source, mock_config),
        get_doc("Чанк A", source, mock_config),
        get_doc("Чанк B", source, mock_config),
    ]
    mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

    with patch(
        "src.RAG.doc_read_service.generate_embeddings_batch"
    ) as mock_batch:
        mock_batch.return_value = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
        result = embed_documents(docs, mock_model)

    assert len(result) == 2
    assert np.array_equal(result[0].embedding, np.array([0.1, 0.2]))
    mock_batch.assert_called_once()


# ─── Тесты read_changed_docs ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_changed_docs_filters_unchanged(
    mock_config: MagicMock,
    tmp_path: Path,
) -> None:
    """read_changed_docs читает только изменённые файлы."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "doc1.txt").write_text(
        "Чанк A.\n\nЧанк B.",
        encoding="utf-8",
    )
    mock_config.DOC_DIR = doc_dir

    with (
        patch(
            "src.RAG.doc_read_service.get_file_type_async",
            new_callable=AsyncMock,
            return_value="text/plain",
        ),
        patch(
            "src.RAG.doc_read_service.get_file_content",
            new_callable=AsyncMock,
            return_value="Чанк A.\n\nЧанк B.",
        ),
        patch(
            "src.RAG.doc_read_service.is_source_changed",
            return_value=False,
        ),
    ):
        docs, changed_sources, manifest = await read_changed_docs(mock_config)

    assert docs == []
    assert changed_sources == []


@pytest.mark.asyncio
async def test_read_changed_docs_returns_changed(
    mock_config: MagicMock,
    tmp_path: Path,
) -> None:
    """read_changed_docs возвращает документы изменённых файлов."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "doc1.txt").write_text(
        "Чанк A.\n\nЧанк B.",
        encoding="utf-8",
    )
    mock_config.DOC_DIR = doc_dir

    with (
        patch(
            "src.RAG.doc_read_service.get_file_type_async",
            new_callable=AsyncMock,
            return_value="text/plain",
        ),
        patch(
            "src.RAG.doc_read_service.get_file_content",
            new_callable=AsyncMock,
            return_value="Чанк A.\n\nЧанк B.",
        ),
        patch(
            "src.RAG.doc_read_service.is_source_changed",
            return_value=True,
        ),
    ):
        docs, changed_sources, manifest = await read_changed_docs(mock_config)

    assert len(docs) == 2
    assert len(changed_sources) == 1


# ─── Тесты read_all_docs ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_all_docs_raises_when_dir_unavailable(
    mock_config: MagicMock,
    tmp_path: Path,
) -> None:
    """read_all_docs выбрасывает NotADirectoryError при недоступной папке."""
    mock_config.DOC_DIR = tmp_path / "does_not_exist"

    with pytest.raises(NotADirectoryError):
        await read_all_docs(mock_config)


@pytest.mark.asyncio
async def test_read_all_docs_returns_sequence(
    mock_config: MagicMock,
    tmp_path: Path,
) -> None:
    """read_all_docs возвращает Sequence[DocumentProtocol]."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "doc1.txt").write_text(
        "Чанк A.\n\nЧанк B.",
        encoding="utf-8",
    )

    mock_config.DOC_DIR = doc_dir

    with (
        patch(
            "src.RAG.doc_read_service.get_file_type_async",
            new_callable=AsyncMock,
            return_value="text/plain",
        ),
        patch(
            "src.RAG.doc_read_service.get_file_content",
            new_callable=AsyncMock,
            return_value="Чанк A.\n\nЧанк B.",
        ),
    ):
        docs = await read_all_docs(mock_config)

    assert isinstance(docs, Sequence)
    assert len(docs) > 0
    assert all(isinstance(doc, Document) for doc in docs)


@pytest.mark.asyncio
async def test_read_all_docs_raises_on_read_error(
    mock_config: MagicMock,
    tmp_path: Path,
) -> None:
    """read_all_docs пробрасывает исключение при ошибке чтения файлов."""
    doc_dir = tmp_path / "docs"
    doc_dir.mkdir()
    (doc_dir / "doc1.txt").write_text("Чанк.", encoding="utf-8")

    mock_config.DOC_DIR = doc_dir

    async def fail_read_file(*args, **kwargs):
        raise RuntimeError("Ошибка чтения файла")

    with patch(
        "src.RAG.doc_read_service.read_file",
        new=fail_read_file,
    ):
        with pytest.raises(RuntimeError, match="Ошибка чтения файла"):
            await read_all_docs(mock_config)
