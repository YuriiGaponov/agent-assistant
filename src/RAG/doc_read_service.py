"""src.RAG.doc_read_service

Интеграция сервисов чтения внутренней документации,
создание структуры данных прочитанных документов
для передачи в другие сервисы.
"""

import asyncio
import hashlib
from datetime import datetime
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np

from src.config import Config
from src.logger import logger
from src.interfaces import DocumentProtocol, EmbeddingModelProtocol
from src.RAG.doc_dir_service import doc_dir_exist, get_doc_files
from src.RAG.doc_file_service import get_file_content
from src.RAG.chunking_service import get_chunks_async
from src.RAG.embedding_service import generate_embeddings_batch
from src.RAG.index_manifest import load_manifest, is_source_changed
from src.RAG.safety import (
    get_file_type_async, UnsupportedFileTypeError, InvalidFileContentError
)
from src.RAG.schemas import Document, DocumentMeta, Source


async def read_file(
    file_path: Path,
    config: Config
) -> list[Document] | None:
    """На основе файла-источника внутренней документации,
    создает список документов без эмбеддингов.

    Выполняет следующие шаги:
      1. Получает MIME-тип файла через get_file_type_async.
      2. Получает содержимое файла через get_file_content.
      3. Вызывает функцию создания структуры данных файла-источника get_source.
      4. Вызывает функцию создания чанков из содержимого файла
         get_chunks_async.
      5. Создает список Document для каждого чанка (без эмбеддингов —
         они генерируются отдельно на этапе пакетной векторизации).

    Args:
        file_path: Путь к файлу-источнику внутренней документации.
        config: Конфигурация приложения (используется для определения
            поддерживаемых типов файлов).

    Returns:
        list[Document]: Список моделей Document по каждому чанку файла,
            либо None, если файл имеет неподдерживаемый тип, содержит
            невалидное содержимое или для его типа отсутствует обработчик.

    Raises:
        UnsupportedFileTypeError: Если тип файла не поддерживается.
        InvalidFileContentError: Если содержимое файла невалидно.
        ValueError: Если отсутствует обработчик для данного типа файла.
    """
    # Проверка типа файла-источника
    try:
        file_type = await get_file_type_async(file_path, config)
    except UnsupportedFileTypeError:
        return None
    except InvalidFileContentError:
        return None

    # Чтение содержимого файла-источника
    try:
        content = await get_file_content(file_path, file_type)
    except ValueError:
        logger.error(
            f'Отсутствует обработчик файлов типа {file_type}',
            file_path=str(file_path.relative_to(config.DOC_DIR))
        )
        return None

    # Создание Source
    source = get_source(file_path, file_type, content)

    # Создание списка Document
    chunks = await get_chunks_async(content)
    docs = [get_doc(chunk, source, config) for chunk in chunks]

    return docs


def hashed(text: str) -> str:
    """Возвращает хэш строки."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_doc(
    chunk: str,
    source: Source,
    config: Config
) -> Document:
    """Создает Document на основе чанка файла-источника.

    Выполняет следующие шаги:
      1. Вычисляет хэш чанка через hashed.
      2. Формирует идентификатор документа как первые 12 символов хэша чанка.
      3. Собирает объект Document с метаданными в формате DocumentMeta.

    Эмбеддинг на этом этапе не генерируется (None) и заполняется позднее
    при пакетной векторизации всех чанков корпуса.

    Сохраняет:
    - id - хэш текста чанка и id файла-источника
    - хэш чанка
    - текст чанка
    - метаданные:
        - модель эмбеддингов
        - данные источника в формате Source

    Args:
        chunk: Текст фрагмента (чанка) документа.
        source: Структура данных файла-источника в формате Source.
        config: Конфигурация приложения, определяющая модель эмбеддингов.

    Returns:
        Document: Модель документа с чанком и метаданными
        (без эмбеддинга).
    """
    chunk_hash = hashed(chunk)
    id = f'doc_{chunk_hash}_src_{source.id}'

    return Document(
            id=id,
            chunk=chunk,
            chunk_hash=chunk_hash,
            meta=DocumentMeta(
                embedding_model=config.EMBEDDING_MODEL,
                source=source
            )
        )


def get_source(file_path: Path, file_type: str, content: str) -> Source:
    """Создание структуры данных файла-источника документов.

    Выполняет следующие шаги:
      1. Вычисляет хэш содержимого файла через hashed.
      2. Формирует идентификатор источника как первые 12 символов хэша.
      3. Определяет дату последнего изменения файла.
      4. Возвращает объект Source с метаданными файла.

    Содержит:
    - id хэш контента
    - путь к файлу
    - тип файла
    - хэш содержимого
    - дата последнего изменения файла

    Args:
        file_path: Путь к файлу-источнику.
        file_type: MIME-тип файла.
        content: Содержимое файла.

    Returns:
        Source: Структура данных файла-источника документов.
    """
    content_hash = hashed(content)
    id = content_hash
    return Source(
        id=id,
        path=file_path,
        type=file_type,
        content_hash=content_hash,
        updated_at=datetime.fromtimestamp(file_path.stat().st_mtime)
    )


def _deduplicate_docs(docs: Sequence[DocumentProtocol]) -> list[DocumentProtocol]:
    """Удаляет дубликаты документов по chunk_hash.

    Идентичные чанки (одинаковый текст) встречаются в корпусе несколько раз
    (например, одинаковые абзацы в разных файлах). Для экономии вычислений
    и устранения дублей записей в векторной БД оставляем первое вхождение
    каждого chunk_hash.

    Args:
        docs: Список документов (чанков) без эмбеддингов.

    Returns:
        list[DocumentProtocol]: Список уникальных по chunk_hash документов.
    """
    unique: dict[str, DocumentProtocol] = {}
    for doc in docs:
        if doc.chunk_hash not in unique:
            unique[doc.chunk_hash] = doc
    return list(unique.values())


def attach_embeddings(
    docs: Sequence[DocumentProtocol],
    embeddings: Sequence[np.ndarray]
) -> list[DocumentProtocol]:
    """Раскладывает эмбеддинги обратно по документам.

    Заполняет поле embedding каждого документа соответствующим вектором
    из списка embeddings (порядок должен совпадать с порядком чанков,
    для которых были сгенерированы эмбеддинги).

    Args:
        docs: Список документов без эмбеддингов.
        embeddings: Список эмбеддингов, по одному на документ.

    Returns:
        list[DocumentProtocol]: Список документов с заполненными эмбеддингами.

    Raises:
        ValueError: Если число эмбеддингов не совпадает с числом документов.
    """
    if len(docs) != len(embeddings):
        raise ValueError(
            'Число эмбеддингов не совпадает с числом документов: '
            f'{len(docs)} != {len(embeddings)}'
        )

    result: list[DocumentProtocol] = []
    for doc, emb in zip(docs, embeddings):
        doc.embedding = emb
        result.append(doc)
    return result


def embed_documents(
    docs: Sequence[DocumentProtocol],
    model: EmbeddingModelProtocol
) -> list[DocumentProtocol]:
    """Дедуплицирует документы и генерирует эмбеддинги пакетно.

    Выполняет следующие шаги:
      1. Удаляет дубликаты чанков по chunk_hash.
      2. Собирает текст всех уникальных чанков.
      3. Генерирует эмбеддинги одним пакетным вызовом.
      4. Раскладывает эмбеддинги обратно по документам.

    Args:
        docs: Список документов без эмбеддингов.
        model: Модель эмбеддингов для генерации векторных представлений.

    Returns:
        list[DocumentProtocol]: Список уникальных документов с заполненными
        эмбеддингами.
    """
    unique_docs = _deduplicate_docs(docs)
    logger.debug(
        'дедупликация чанков завершена',
        total=len(docs),
        unique=len(unique_docs)
    )
    chunks = [doc.chunk for doc in unique_docs]
    embeddings = generate_embeddings_batch(model, chunks)
    return attach_embeddings(unique_docs, embeddings)


async def read_all_docs_with_embeddings(
    config: Config,
    model: EmbeddingModelProtocol
) -> Sequence[DocumentProtocol]:
    """Читает все документы из директории и генерирует эмбеддинги пакетно.

    Этапы:
      1. Проверяет директорию и получает список файлов.
      2. Параллельно читает файлы и разбивает их на чанки (без эмбеддингов).
      3. Удаляет дубликаты чанков по chunk_hash.
      4. Генерирует эмбеддинги одним пакетным вызовом.
      5. Раскладывает эмбеддинги обратно по документам.

    Args:
        config: Конфигурация с путём к директории документов.
        model: Модель эмбеддингов для генерации векторных представлений.

    Returns:
        Sequence[DocumentProtocol]: Список документов с чанками, эмбеддингами
        и метаданными.

    Raises:
        NotADirectoryError: Если директория не существует или не является
        папкой.
    """
    docs = await read_all_docs(config)
    return embed_documents(docs, model)


async def read_all_docs(
    config: Config,
) -> Sequence[DocumentProtocol]:
    """Читает все документы из указанной директории и возвращает список
    моделей Document (без эмбеддингов).

    Выполняет следующие шаги:
      1. Проверяет существование и корректность директории через doc_dir_exist.
      2. Получает список файлов через get_doc_files.
      3. Параллельно читает содержимое файлов и разбивает на чанки через
         read_file.
      4. Возвращает список объектов Document для дальнейшей обработки в
         RAG‑конвейере.

    Args:
        config: Конфигурация с путём к директории документов (config.DOC_DIR).

    Returns:
        list[Document]: Список моделей Document с содержимым и чанками
        (без эмбеддингов).

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
    docs = []
    docs_ids = set()
    # docs = {}
    try:
        logger.debug('Чтение файлов из списка, создание списка Document...')
        read_tasks = [read_file(path, config) for path in doc_paths]
        readed_docs = await asyncio.gather(*read_tasks)
        for list_doc in readed_docs:
            if list_doc is not None:
                # docs.extend(doc)
                for doc in list_doc:
                    doc_id = doc.id
                    if doc_id not in docs_ids:
                        docs_ids.add(doc_id)
                        docs.append(doc)
    except Exception as e:
        logger.critical(f'ошибка чтения файлов: {e}')
        raise
    return docs


async def read_changed_docs(
    config: Config,
) -> tuple[list[DocumentProtocol], list[Source], dict[str, dict[str, Any]]]:
    """Инкрементально читает только изменённые и новые файлы.

    Сравнивает состояние файлов-источников с манифестом индексации
    и читает (чанкает) лишь те, чьё содержимое изменилось или которые
    отсутствуют в манифесте.

    Args:
        config: Конфигурация приложения.

    Returns:
        tuple:
            - список документов (чанков без эмбеддингов) изменённых файлов;
            - список источников изменённых файлов;
            - актуальный манифест индексации.
    """
    logger.debug('Проверка валидности папки с документацией...')
    if not doc_dir_exist(config):
        raise NotADirectoryError('Папка внутренней документации недоступна')

    manifest = load_manifest(config)
    doc_paths = get_doc_files(config)

    # Читаем файлы параллельно, получаем список документов по каждому файлу.
    read_tasks = [read_file(path, config) for path in doc_paths]
    readed_docs = await asyncio.gather(*read_tasks)

    docs: list[DocumentProtocol] = []
    changed_sources: list[Source] = []
    for file_docs in readed_docs:
        if file_docs is None:
            continue
        # Источник берём из первого документа файла.
        source = file_docs[0].meta.source
        if not is_source_changed(config, source):
            continue
        docs.extend(file_docs)
        changed_sources.append(source)

    return docs, changed_sources, manifest
