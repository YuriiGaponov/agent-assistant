"""main.py

Точка входа FastAPI‑приложения.
Инициализирует экземпляр приложения app и служит основой для регистрации
роутеров и настройки зависимостей.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src import (
    get_config, logger, router, embedding_model, init_vector_collection,
    get_vectorDB_manager
)

config = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info(f'Запуск {app.title}')
        app.state.config = config
        app.state.embedding_model = embedding_model
        logger.debug('инициализация менеджера векторной БД...')
        vectorDB = get_vectorDB_manager(config)
        app.state.vectorDB = vectorDB
        logger.debug('инициализация векторной коллекции...')
        await init_vector_collection(app)
        logger.info(f'приложение {app.title} запущено')
        yield
    except Exception as e:
        logger.critical(f'Ошибка запуска: {e}')
        raise
    finally:
        logger.info(f'Остановка {app.title}')

app = FastAPI(
    debug=config.DEBUG,
    title=config.TITLE,
    description=config.DESCRIPTION,
    version=config.VERSION,
    lifespan=lifespan
)

app.include_router(router)
