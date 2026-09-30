"""src.models.user

ORM-модель пользователя для БД проекта Agent-Assistant.
Объединяет поля аутентификации (fastapi-users) и асинхронную базу (Base).
Позволяет расширять модель кастомными полями без правки логики auth.
Участвует в миграциях Alembic через Base.metadata.
Совместима с async_sessionmaker и FastAPI-сервисами.
"""

from fastapi_users.db import SQLAlchemyBaseUserTable

from src.models.base import Base


class UserModel(SQLAlchemyBaseUserTable, Base):
    """Модель пользователя, совместимая с fastapi-users и async SQLAlchemy.

    Наследуется от:
      - SQLAlchemyBaseUserTable: базовые поля для auth.
      - Base: регистрация в метаданных и async-поддержка.
    """
    __tablename__ = "users"
