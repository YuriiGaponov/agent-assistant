"""src.users.schemas

Модели пользователей.
"""


from fastapi_users.schemas import BaseUser, BaseUserCreate, BaseUserUpdate


class UserRead(BaseUser[int]):
    """Основная модель пользователя."""


class UserCreate(BaseUserCreate):
    """Основная модель создания пользователя."""


class UserUpdate(BaseUserUpdate):
    """Основная модель редактирования пользователя."""
