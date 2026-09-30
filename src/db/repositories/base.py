"""src.db.repositories.base

Базовый класс для всех репозиториев реляционной БД в проекте Agent‑Assistant.
"""

from abc import ABC

from sqlalchemy.ext.asyncio import AsyncSession


class BaseRepository(ABC):
    """
    Базовый абстрактный репозиторий для работы с сущностями в реляционной БД.

    Этот класс не привязан к конкретной модели: он задаёт контракт и общую
    логику, которую можно переиспользовать во всём проекте.
    """

    def __init__(self, db: AsyncSession):
        """
        Инициализирует репозиторий активной сессией БД.

        Args:
            db: Активная AsyncSession для выполнения запросов.
        """
        self.db = db
