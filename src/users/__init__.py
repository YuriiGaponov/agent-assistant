"""src.users.__init__

Пакет реализации логики работы с данными пользователей.
"""

from src.users.authentication import auth_backend
from src.users.dependencies import get_current_user, fastapi_users
from src.users.schemas import UserCreate, UserRead

__all__ = ['get_current_user', 'fastapi_users', 'auth_backend',
           'UserCreate', 'UserRead']
