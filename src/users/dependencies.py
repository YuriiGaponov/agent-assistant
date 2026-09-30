"""src.users.dependencies

Модуль зависимостей, реализующих взаимодействие с пользователями.
"""

from typing import AsyncGenerator

from fastapi import Depends
from fastapi_users import FastAPIUsers
from fastapi_users.db import SQLAlchemyUserDatabase
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import get_db
from src.models import UserModel
from src.users.authentication import auth_backend
from src.users.user_manager import UserManager


async def get_user_db(
    session: AsyncSession = Depends(get_db)
) -> AsyncGenerator[SQLAlchemyUserDatabase, None]:
    """Фабрика адаптеров базы данных для работы с пользователями.

    Создаёт экземпляр SQLAlchemyUserDatabase на основе сессии из get_db.
    Используется fastapi-users как зависимость user_db.

    Yields:
        SQLAlchemyUserDatabase: экземпляр базы пользователей для fastapi-users.
    """
    yield SQLAlchemyUserDatabase(session, UserModel)


async def get_user_manager(user_db=Depends(get_user_db)):
    """Фабрика менеджера пользователей.

    Создаёт экземпляр UserManager на основе user_db из fastapi-users.
    Позволяет использовать методы регистрации, сброса пароля и др.

    Yields:
        UserManager: экземпляр менеджера пользователей.
    """
    yield UserManager(user_db)


fastapi_users = FastAPIUsers[UserModel, int](
    get_user_manager,
    [auth_backend],
)
"""
Экземпляр FastAPIUsers для управления аутентификацией.

Объединяет UserManager и бэкенды аутентификации, генерирует API‑маршруты.
"""

get_current_user = fastapi_users.current_user()
"""Фабрика получения текущего пользователя из запроса."""
