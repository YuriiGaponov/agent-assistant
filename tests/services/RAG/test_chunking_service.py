"""tests.RAG.test_chunking_service

Тесты разбивки текста на смысловые чанки для RAG‑конвейера.
Проверяют синхронную функцию get_chunks и асинхронную обёртку
get_chunks_async.
"""

import pytest

from src.RAG.chunking_service import get_chunks, get_chunks_async


# ─── Синхронные тесты get_chunks ────────────────────────────────────────


def test_get_chunks_splits_by_paragraphs() -> None:
    """Разбиение по двойным переносам строк на отдельные абзацы."""
    content = "Первый абзац.\n\nВторой абзац.\n\nТретий абзац."
    chunks = get_chunks(content)
    assert len(chunks) == 3
    assert chunks[0] == "Первый абзац."
    assert chunks[1] == "Второй абзац."
    assert chunks[2] == "Третий абзац."


def test_get_chunks_ignores_empty_blocks() -> None:
    """Игнорирование множественных пустых строк между абзацами."""
    content = "Текст.\n\n\n\nЕщё текст."
    chunks = get_chunks(content)
    assert len(chunks) == 2
    assert chunks[0] == "Текст."
    assert chunks[1] == "Ещё текст."


def test_get_chunks_handles_single_paragraph() -> None:
    """Обработка текста с единственным абзацем."""
    content = "Единственный абзац."
    chunks = get_chunks(content)
    assert len(chunks) == 1
    assert chunks[0] == "Единственный абзац."


def test_get_chunks_returns_empty_list_for_empty_string() -> None:
    """Пустая строка должна возвращать пустой список."""
    chunks = get_chunks("")
    assert chunks == []


def test_get_chunks_returns_empty_list_for_whitespace_only() -> None:
    """Строка из пробелов и переносов возвращает пустой список."""
    chunks = get_chunks("   \n\n  \n  ")
    assert chunks == []


def test_get_chunks_strips_whitespace() -> None:
    """Удаление лишних пробелов и переносов �� начале/конце чанка."""
    content = "  Текст с пробелами.  \n\n  Ещё текст.  "
    chunks = get_chunks(content)
    assert len(chunks) == 2
    assert chunks[0] == "Текст с пробелами."
    assert chunks[1] == "Ещё текст."


def test_get_chunks_no_double_newline() -> None:
    """Текст без двойных переносов возвращается как один чанк."""
    content = "Строка1\nСтрока2\nСтрока3"
    chunks = get_chunks(content)
    assert len(chunks) == 1
    assert chunks[0] == content


# ─── Асинхронные тесты get_chunks_async ─────────────────────────────────


@pytest.mark.asyncio
async def test_get_chunks_async_returns_same_as_sync() -> None:
    """Асинхронная обёртка возвращает тот же результат, что и синхронная."""
    content = "Абзац A.\n\nАбзац B.\n\nАбзац C."
    chunks_sync = get_chunks(content)
    chunks_async = await get_chunks_async(content)

    assert chunks_async == chunks_sync


@pytest.mark.asyncio
async def test_get_chunks_async_empty_string() -> None:
    """Асинхронная обёртка корректно обрабатывает пустую строку."""
    result = await get_chunks_async("")
    assert result == []


@pytest.mark.asyncio
async def test_get_chunks_async_single_paragraph() -> None:
    """Асинхронная обёртка корректно обрабатывает один абзац."""
    result = await get_chunks_async("Один абзац.")
    assert len(result) == 1
    assert result[0] == "Один абзац."
