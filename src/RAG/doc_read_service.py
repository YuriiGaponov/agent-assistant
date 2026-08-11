"""src.RAG.doc_read_service

Интеграция сервисов чтения внутренней документации,
создание структуры данных прочитанных документов
для передачи в другие сервисы.
"""

import asyncio
from pathlib import Path

from src.config import Config
from src.logger import logger
from src.interfaces import Document as ReadedDocument, EmbeddingModelProtocol
from src.RAG.doc_dir_service import doc_dir_exist, get_doc_files
from src.RAG.doc_file_service import get_file_content
from src.RAG.chunking_service import get_chunks_async
from src.RAG.embedding_service import generate_embeddings
from src.RAG.schemas import Document


async def read_doc(
    file_path: Path,
    model: EmbeddingModelProtocol
) -> ReadedDocument:
    """Читает документ и возвращает его в виде модели Document.

    Загружает содержимое файла, разбивает его на чанки по абзацам
    и упаковывает в модель Document. Используется как базовый шаг
    подготовки документа для RAG‑конвейера.
    """
    content = await get_file_content(file_path)
    chunks = await get_chunks_async(content)
    embeddings = generate_embeddings(model, chunks)
    return Document(chunks=chunks, embeddings=embeddings)


async def read_all_docs(
    config: Config,
    model: EmbeddingModelProtocol
) -> list[ReadedDocument]:
    """Читает все документы из указанной директории и возвращает список
    моделей Document.

    Выполняет следующие шаги:
      1. Проверяет существование и корректность директории через doc_dir_exist.
      2. Получает список файлов через get_doc_files.
      3. Параллельно читает содержимое файлов и разбивает на чанки через
         read_doc.
      4. Возвращает список объектов Document для дальнейшей обработки в
         RAG‑конвейере.

    Args:
        config: Конфигурация с путём к директории документов (config.DOC_DIR).

    Returns:
        list[Document]: Список моделей Document с содержимым и чанками.

    Raises:
        NotADirectoryError: Если директория не существует или не является
        папкой.
    """
    logger.debug('Проверка валидности папки с документацией...')
    if not doc_dir_exist(config):
        logger.debug(f'DOC_DIR: {config.model_config}')
        raise NotADirectoryError('Папка внутренней документации недоступна')
    logger.debug('Получение списка файлов...')
    doc_paths = get_doc_files(config)
    try:
        logger.debug('Чтение файлов из списка, создание списка Document...')
        read_tasks = [read_doc(path, model) for path in doc_paths]
        readed_docs = await asyncio.gather(*read_tasks)
    except Exception as e:
        logger.critical(f'ошибка чтения файлов: {e}')
        raise
    return readed_docs
