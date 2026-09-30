"""src.db.session

Модуль реализации получения сессий БД.
"""

from sqlalchemy.ext.asyncio import (
    AsyncEngine, async_sessionmaker, AsyncSession
)
from src.logger import logger


def create_session_maker(engine: AsyncEngine) -> async_sessionmaker:
    """
    Создаёт и возвращает фабрику сессий на основе переданного движка.
    Не хранит состояние, только конструирует объект.
    """
    session_maker = async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        class_=AsyncSession,
        future=True,
    )
    logger.debug("Session maker created")
    return session_maker
