"""src.RAG.init_vector_collection

Модуль инициализации векторной коллекции при старте FastAPI‑приложения.
Выполняет полный RAG‑цикл холодного старта: чтение документов → чанкинг →
эмбеддинги → индексация в векторную БД. Предназначен для использования
в lifespan‑контексте приложения (startup).

Поддерживает инкрементальную индексацию на основе манифеста:
повторные запуски обрабатывают только изменённые и новые файлы,
а неизменные пропускаются.
"""

from datetime import datetime

from fastapi import FastAPI

from src.logger import logger
from src.interfaces import VectorDBManagerInterface
from src.RAG.doc_read_service import (
    read_changed_docs, embed_documents
)
from src.RAG.index_manifest import save_manifest, record_source


async def init_vector_collection(app: FastAPI):
    """Инициализирует векторную коллекцию при старте приложения.

    Выполняет следующие шаги:
      1. Получает конфигурацию, модель эмбеддингов и менеджер векторной БД
         из состояния приложения (app.state).
      2. Инкрементально читает только изменённые и новые документы
         из директории внутренней документации.
      3. Для изменённых источников удаляет их старые записи из векторной БД.
      4. Генерирует эмбеддинги пакетно и сохраняет документы в векторную БД.
      5. Обновляет манифест индексации.

    Args:
        app: Экземпляр FastAPI‑приложения, в state которого хранятся
             config, embedding_model и vectorDB.

    Raises:
        Exception: Пробрасывается вверх, если на любом этапе произошла ошибка,
                   чтобы lifespan корректно зафиксировал сбой запуска.
    """
    config = app.state.config
    model = app.state.embedding_model
    vectorDB: VectorDBManagerInterface = app.state.vectorDB
    try:
        start_time = datetime.now()
        logger.debug('чтение изменённой внутренней документации...')
        changed_docs, changed_sources, manifest = await read_changed_docs(config)

        if not changed_docs:
            logger.debug(
                'изменений в документации нет, индексация не требуется',
                index_time=datetime.now()-start_time
            )
            return

        # Удаляем старые записи изменённых источников во избежание дублей.
        for source in changed_sources:
            await vectorDB.delete_by_source(source.id)

        docs_with_embeddings = embed_documents(changed_docs, model)

        logger.debug('сохранение документов...')
        await vectorDB.add(docs_with_embeddings)

        # Обновляем манифест индексации.
        for source in changed_sources:
            record_source(config, manifest, source)
        save_manifest(config, manifest)
    except Exception as e:
        logger.critical(f'ошибка инициализации коллекции: {e}')
        raise
    logger.debug(
        'векторная коллекция создана',
        init_time=datetime.now()-start_time
    )
