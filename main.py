"""main

Точка входа FastAPI‑приложения.
Инициализирует экземпляр приложения app и служит основой для регистрации
роутеров и настройки зависимостей.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src import (
    get_config, logger, router, embedding_model, init_vector_collection,
    get_vectorDB_manager, init_agent, RequestLoggingMiddleware,
    get_async_engine, create_session_maker
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
        logger.debug('инициализация менеджера главной LLM...')
        agent = init_agent(config, embedding_model, vectorDB)
        app.state.agent = agent
        logger.debug('инициализация векторной коллекции...')
        await init_vector_collection(app)
        logger.debug('инициализация движка реляционной БД...')
        async_engine = get_async_engine(config)
        app.state.async_engine = async_engine
        logger.debug('инициализация фабрики сессий реляционной БД...')
        session_maker = create_session_maker(async_engine)
        app.state.session_maker = session_maker
        logger.info(f'приложение {app.title} запущено')
        yield
    except Exception as e:
        logger.critical(f'Ошибка запуска: {e}')
        raise
    finally:
        logger.info(f'Остановка {app.title}')
        # Graceful shutdown: корректно закрываем клиент векторной БД,
        # чтобы выгрузить незавершённые данные (HNSW-индекс) на диск
        # и избежать его повреждения при повторном запуске.
        if vectorDB is not None:
            await vectorDB.close()
        # Закрытие движка реляционной БД (dispose() для async_engine)
        if async_engine is not None:
            await async_engine.dispose()


app = FastAPI(
    debug=config.DEBUG,
    title=config.TITLE,
    description=config.DESCRIPTION,
    version=config.VERSION,
    lifespan=lifespan
)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(router)
