"""tests.RAG.test_doc_file_service

Тесты чтения файлов и метаданных в Agent‑Assistant.
Проверяют корректность асинхронного чтения текстовых файлов
через aiofiles, включая сценарии отсутствия файла, неверной
кодировки, пустого и большого содержимого.
"""

from pathlib import Path

import pytest

from src.RAG.doc_file_service import get_file_content, read_txt_file


@pytest.mark.asyncio
async def test_read_txt_file_success(tmp_path: Path) -> None:
    """Успешное чтение существующего текстового файла."""
    file_path = tmp_path / "test.txt"
    content = "Привет, Agent‑Assistant!\nЭто тестовый файл."
    file_path.write_text(content, encoding="utf-8")

    result = await read_txt_file(file_path, encoding="utf-8")

    assert result == content


@pytest.mark.asyncio
async def test_read_txt_file_not_found() -> None:
    """Чтение несуществующего файла должно выбрасывать FileNotFoundError."""
    non_existent = Path("/non/existent/file.txt")

    with pytest.raises(FileNotFoundError):
        await read_txt_file(non_existent)


@pytest.mark.asyncio
async def test_read_txt_file_invalid_encoding(tmp_path: Path) -> None:
    """Неверная кодировка должна приводить к ошибке декодирования."""
    binary_path = tmp_path / "binary.bin"
    binary_path.write_bytes(b"\xff\xfe\x00\x01")  # невалидный UTF‑8

    with pytest.raises(UnicodeDecodeError):
        await read_txt_file(binary_path, encoding="utf-8")


@pytest.mark.asyncio
async def test_read_txt_file_empty_file(tmp_path: Path) -> None:
    """Чтение пустого файла должно возвращать пустую строку."""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    result = await read_txt_file(file_path)

    assert result == ""


@pytest.mark.asyncio
async def test_read_txt_file_large_content(tmp_path: Path) -> None:
    """Проверка чтения большого текста без потери данных."""
    file_path = tmp_path / "large.txt"
    content = "x" * 100_000
    file_path.write_text(content, encoding="utf-8")

    result = await read_txt_file(file_path)

    assert len(result) == 100_000
    assert result == content


@pytest.mark.asyncio
async def test_read_txt_file_respects_encoding(tmp_path: Path) -> None:
    """Убеждаемся, что кодировка применяется корректно."""
    file_path = tmp_path / "cyrillic.txt"
    content = "Русский текст с символами: ё, ъ, ы, ь"
    file_path.write_text(content, encoding="utf-8")

    result = await read_txt_file(file_path, encoding="utf-8")

    assert result == content
    assert "ё" in result


@pytest.mark.asyncio
async def test_get_file_content_delegates_to_read_txt_file(tmp_path: Path) -> None:
    """get_file_content делегирует чтение read_txt_file и возвращает содержимое."""
    file_path = tmp_path / "delegate.txt"
    content = "Делегированный вызов."
    file_path.write_text(content, encoding="utf-8")

    result = await get_file_content(file_path)

    assert result == content


@pytest.mark.asyncio
async def test_get_file_content_passes_encoding(tmp_path: Path) -> None:
    """get_file_content передаёт параметр encoding в read_txt_file."""
    file_path = tmp_path / "encoding_test.txt"
    content = "Текст в UTF-8"
    file_path.write_text(content, encoding="utf-8")

    result = await get_file_content(file_path, encoding="utf-8")

    assert result == content


@pytest.mark.asyncio
async def test_get_file_content_not_found() -> None:
    """get_file_content пробрасывает FileNotFoundError от read_txt_file."""
    non_existent = Path("/non/existent/file.txt")

    with pytest.raises(FileNotFoundError):
        await get_file_content(non_existent)
