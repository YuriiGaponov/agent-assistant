"""src.db.engine

Модуль управления движком реляционной БД.
"""

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from src.config import Config
from src.logger import logger


def get_async_engine(config: Config) -> AsyncEngine:
    """
    Создаёт асинхронный движок SQLAlchemy для реляционной БД.

    Args:
        config: Экземпляр конфигурации с полем DATABASE_URL.

    Returns:
        AsyncEngine: Настроенный движок для работы с БД.

    Raises:
        Exception: Пробрасывается дальше, если инициализация не удалась.
    """
    try:
        async_engine = create_async_engine(config.DATABASE_URL)
        logger.debug('инициализирован движок реляционной БД.')
        return async_engine
    except Exception as e:
        logger.error('ошибка инициализации движка реляционной БД', error=e)
        raise
