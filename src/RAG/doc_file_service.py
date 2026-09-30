"""src.RAG.doc_file_service

Модуль, реализующий логику чтения файлов для Agent‑Assistant.
Содержит асинхронные функции для загрузки содержимого документов.
На текущем этапе поддерживает чтение текстовых файлов с помощью
aiofiles; в дальнейшем планируется расширение для PDF, DOCX
и других форматов. Используется в рамках RAG‑цикла для извлечения
контента перед разбиением на чанки.
"""

import aiofiles
from pathlib import Path


async def get_file_content(
    file_path: Path,
    file_type: str,
    encoding: str = "utf-8"
) -> str:
    """Читает содержимое файла через специализированную функцию.

    Делегирует чтение файла функции read_txt_file. На текущем этапе
    поддерживает только текстовые файлы; в будущем будет расширен
    логикой выбора парсера в зависимости от расширения файла.
    Возвращает содержимое файла как строку.
    """
    if file_type == "text/plain":
        return await read_txt_file(file_path, encoding)
    else:
        raise ValueError(
            f'Отсутствует обработчик файлов типа {file_type}'
        )


async def read_txt_file(file_path: Path,  encoding: str = "utf-8") -> str:
    """Асинхронно читает содержимое текстового файла.

    Использует aiofiles для неблокирующего чтения файла в режиме
    чтения с кодировкой UTF‑8. Возвращает содержимое файла в виде
    строки. Обеспечивает корректную работу в асинхронном контексте
    FastAPI и позволяет избежать блокировки event loop при работе
    с файловой системой.
    """
    async with aiofiles.open(file_path, "r", encoding=encoding) as f:
        return await f.read()
