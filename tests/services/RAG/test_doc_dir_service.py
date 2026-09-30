"""tests.RAG.test_doc_dir_service

Тесты логики взаимодействия с директорией внутренней
документации в Agent‑Assistant.
Проверяют корректность проверки существования директории
для хранения документов, включая сценарии отсутствия папки
и указания файла вместо директории.
"""

from pathlib import Path
import tempfile

from src.config import Config
from src.RAG.doc_dir_service import doc_dir_exist, get_doc_files


def test_doc_dir_exist_returns_false_when_missing() -> None:
    """Проверяет возврат False, если директория не существует.

    Создаёт путь к несуществующей подпапке внутри временной директории
    и убеждается, что doc_dir_exist корректно возвращает False.
    Тестирует базовую валидацию пути перед операциями с документами.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        nonexistent_path = Path(tmpdir) / "does_not_exist"
        config = Config(DOC_DIR=nonexistent_path)
        assert doc_dir_exist(config) is False


def test_doc_dir_exist_returns_false_when_file() -> None:
    """Проверяет возврат False, если путь указывает на файл.

    Создаёт временный файл и передаёт его путь в config.DOC_DIR.
    Убеждается, что doc_dir_exist распознаёт файл как недопустимую
    директорию и возвращает False вместо выброса исключения.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "file.txt"
        file_path.write_text("dummy")
        config = Config(DOC_DIR=file_path)
        assert doc_dir_exist(config) is False


def test_doc_dir_exist_returns_true_when_valid() -> None:
    """Проверяет возврат True при корректной директории.

    Передаёт в функцию существующую временную директорию и убеждается,
    что doc_dir_exist возвращает True. Подтверждает корректное поведение
    в «счастливом» сценарии, когда DOC_DIR настроен верно.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(DOC_DIR=Path(tmpdir))
        assert doc_dir_exist(config) is True


def test_get_doc_files_returns_empty_list_when_directory_is_empty() -> None:
    """Проверяет, что при пустой директории возвращается пустой список.

    Создаёт временную пустую директорию, передаёт её в config.DOC_DIR
    и убеждается, что get_doc_files возвращает пустой список.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        config = Config(DOC_DIR=Path(tmpdir))
        result = get_doc_files(config)
        assert result == []


def test_get_doc_files_returns_files_in_directory() -> None:
    """Проверяет возврат списка файлов при наличии файлов в директории.

    Создаёт несколько файлов в временной директории и убеждается,
    что get_doc_files корректно возвращает их как объекты Path.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        file1 = dir_path / "file1.txt"
        file2 = dir_path / "report.pdf"
        file3 = dir_path / "data.csv"

        # Создаём файлы
        file1.write_text("content1")
        file2.write_text("content2")
        file3.write_text("content3")

        config = Config(DOC_DIR=dir_path)
        result = get_doc_files(config)

        assert len(result) == 3
        assert file1 in result
        assert file2 in result
        assert file3 in result

        # Убеждаемся, что все элементы — это Path
        for path in result:
            assert isinstance(path, Path)
