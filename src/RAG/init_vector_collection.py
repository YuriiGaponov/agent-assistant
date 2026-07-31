"""src.RAG.init_vector_collection

Модуль инициализации векторной коллекции при старте FastAPI‑приложения.
Выполняет полный RAG‑цикл холодного старта: чтение документов → чанкинг →
эмбеддинги → индексация в векторную БД. Предназначен для использования
в lifespan‑контексте приложения (startup).
"""

from fastapi import FastAPI

from src.logger import logger
from src.interfaces import VectorDBManagerInterface
from src.RAG.doc_read_service import read_all_docs


async def init_vector_collection(app: FastAPI):
    """Инициализирует векторную коллекцию при старте приложения.

    Выполняет следующие шаги:
      1. Получает конфигурацию, модель эмбеддингов и менеджер векторной БД
         из состояния приложения (app.state).
      2. Читает все документы из директории внутренней документации.
      3. Сохраняет обработанные документы (чанки + эмбеддинги) в векторную БД
         через интерфейс VectorDBManagerInterface.

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
        logger.debug('чтение внутренней документации...')
        readed_docs = await read_all_docs(config, model)
        logger.debug('сохранение документов...')
        await vectorDB.add(readed_docs)
    except Exception as e:
        logger.critical(f'ошибка инициализации коллекции: {e}')
        raise
    logger.debug('векторная коллекция создана')
