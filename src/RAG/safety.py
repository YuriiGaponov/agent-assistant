"""src.RAG.safety

Сервис обеспечения безопасной индексации файлов внутренней документации.

Предоставляет:
    - Проверку типа файла по расширению и MIME (через python-magic).
    - Валидацию соответствия расширения и реального содержимого.
    - Защиту от обработки подозрительных файлов
      (подмена расширений, битые файлы).
    - Чёткие исключения для оркестрации в вышестоящих сервисах
      (чтобы не ломать весь пайплайн).
"""

from pathlib import Path

import magic

from src.config import Config, SUPPORTED_DOC_TYPES
from src.logger import logger


class UnsupportedFileTypeError(Exception):
    """Файл имеет неподдерживаемое расширение и не должен обрабатываться."""
    pass


class InvalidFileContentError(Exception):
    """Содержимое файла не соответствует ожидаемому типу (MIME/структура)."""
    pass


def get_file_type(
    file_path: Path,
    config: Config,
) -> str:
    """
    Проверяет тип файла по расширению и MIME.

    Выполняет три уровня проверки:
        1. Расширение файла должно быть в SUPPORTED_DOC_TYPES.
        2. MIME-тип файла должен быть в разрешённых значениях.
        3. MIME-тип должен соответствовать ожидаемому для данного расширения.

    Args:
        file_path: Путь к файлу.
        config: Конфигурация приложения
                (для DOC_DIR и списка поддерживаемых типов).

    Returns:
        str: MIME-тип файла, если все проверки пройдены.

    Raises:
        UnsupportedFileTypeError: Если расширение не поддерживается.
        InvalidFileContentError: Если MIME-тип не поддерживается
                                 или не соответствует расширению.
    """
    # Проверка типа файла по расширению
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_DOC_TYPES:
        logger.warning(
            f'неподдерживаемый тип файла {file_path.name}',
            file_path=file_path.relative_to(config.DOC_DIR)
        )
        raise UnsupportedFileTypeError(
            f"Неподдерживаемое расширение {ext} для файла {file_path}"
        )

    # Проверка MIME
    try:
        mime = magic.from_file(str(file_path), mime=True)
    except Exception as e:
        logger.error(
            f"ошибка определения MIME-типа для {file_path.name}: {e}",
            file_path=str(file_path.relative_to(config.DOC_DIR)),
        )
        raise InvalidFileContentError(
            f"Не удалось определить MIME-тип для файла {file_path}: {e}"
        ) from e

    if mime not in SUPPORTED_DOC_TYPES.values():
        logger.warning(
            f'неподдерживаемый MIME-тип файла {file_path.name}',
            file_path=file_path.relative_to(config.DOC_DIR)
        )
        raise InvalidFileContentError(
            f'неподдерживаемый MIME-тип файла {file_path.name}'
        )

    # Проверка соответствия типа сдержимого и расширения
    expected_mime = SUPPORTED_DOC_TYPES.get(ext)
    if mime != expected_mime:
        logger.warning(
            f"MIME-тип {file_path.name} не соответствует расширению {ext}",
            file_path=str(file_path.relative_to(config.DOC_DIR)),
            mime=mime,
            expected_mime=expected_mime,
            ext=ext
        )
        raise InvalidFileContentError(
            f'несоответствие MIME-типа расширению файла {file_path.name}'
        )

    return mime


async def get_file_type_async(
    file_path: Path,
    config: Config,
) -> str:
    """
    Асинхронная обёртка для проверки файла.

    В текущей реализации get_file_type синхронный (magic — это C-вызов),
    поэтому мы запускаем его в executor, чтобы не блокировать event loop.

    Args:
        file_path: Путь к файлу.
        config: Конфигурация.

    Returns:
        MIME-тип файла.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, get_file_type, file_path, config)
