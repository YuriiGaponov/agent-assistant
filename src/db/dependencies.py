"""src.db.dependencies

Зависимости для работы с БД в FastAPI-приложении.

Предоставляет:
    - Зависимость get_db для внедрения сессии БД в эндпоинты.
    - Безопасную работу с session_maker из app.state.
"""

from typing import AsyncGenerator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """
    Зависимость для получения сессии БД.

    Извлекает session_maker из app.state, создаёт сессию и отдаёт её в
    эндпоинт.
    Сессия автоматически закрывается (commit/rollback) при выходе из контекста.

    Args:
        request: Объект запроса FastAPI (нужен для доступа к app.state).

    Yields:
        AsyncSession: Активная сессия для работы с БД.

    Raises:
        RuntimeError: Если session_maker не найден в app.state
                      (ошибка инициализации приложения).
    """
    session_maker = request.app.state.session_maker
    async with session_maker() as session:
        yield session
