"""tests.RAG.test_safety

Тесты проверки типа файла в Agent‑Assistant.
Проверяют корректность работы :func:`src.RAG.safety.get_file_type`:
возврат допустимого MIME‑типа, обработку недопустимого расширения,
ошибки определения MIME, неподдерживаемого MIME и несоответствия
MIME‑типа расширению.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.RAG.safety import (
    get_file_type,
    InvalidFileContentError,
    UnsupportedFileTypeError,
)


def test_get_file_type_returns_valid_mime(test_doc_file_path: Path,
                                          test_doc_config) -> None:
    """Возврат допустимого MIME‑типа при успешной проверке.

    Для файла с расширением ``.txt`` и MIME ``text/plain`` функция
    должна вернуть ``text/plain``.
    """
    with patch("src.RAG.safety.magic.from_file",
               return_value="text/plain"):
        result = get_file_type(test_doc_file_path, test_doc_config)

    assert result == "text/plain"


def test_get_file_type_unsupported_extension(test_doc_file_path: Path,
                                             test_doc_config) -> None:
    """Недопустимое расширение должно вызывать UnsupportedFileTypeError.

    Файл с расширением, отсутствующим в SUPPORTED_DOC_TYPES
    (например, ``.pdf``), должен отклоняться до обращения к magic.
    """
    bad_path = test_doc_file_path.with_suffix(".pdf")

    with patch("src.RAG.safety.magic.from_file") as mock_magic:
        with pytest.raises(UnsupportedFileTypeError):
            get_file_type(bad_path, test_doc_config)

    mock_magic.assert_not_called()


def test_get_file_type_mime_error(test_doc_file_path: Path,
                                  test_doc_config) -> None:
    """Ошибка получения MIME‑типа должна вызывать InvalidFileContentError.

    При исключении внутри magic.from_file функция должна пробросить
    InvalidFileContentError.
    """
    with patch("src.RAG.safety.magic.from_file",
               side_effect=RuntimeError("magic failed")):
        with pytest.raises(InvalidFileContentError):
            get_file_type(test_doc_file_path, test_doc_config)


def test_get_file_type_unsupported_mime(test_doc_file_path: Path,
                                        test_doc_config) -> None:
    """Неподдерживаемый MIME‑тип должен вызывать InvalidFileContentError.

    Если magic вернул MIME, отсутствующий в разрешённых значениях
    (например, ``application/octet-stream``), функция должна
    отклонить файл.
    """
    with patch("src.RAG.safety.magic.from_file",
               return_value="application/octet-stream"):
        with pytest.raises(InvalidFileContentError):
            get_file_type(test_doc_file_path, test_doc_config)


def test_get_file_type_mime_mismatch(test_doc_file_path: Path,
                                     test_doc_config) -> None:
    """Несоответствие MIME расширению должно вызывать InvalidFileContentError.

    Для файла ``.txt``, которому соответствует ``text/plain``, возврат
    другого (но допустимого в целом) MIME должен привести к ошибке.
    """
    with patch("src.RAG.safety.magic.from_file",
               return_value="application/json"):
        with pytest.raises(InvalidFileContentError):
            get_file_type(test_doc_file_path, test_doc_config)
