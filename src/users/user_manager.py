"""src.users.user_manager

Реализация основной логики работы с пользователями.
"""

from fastapi_users import BaseUserManager, IntegerIDMixin

from src.config import get_config
from src.models import UserModel

config = get_config()


class UserManager(IntegerIDMixin, BaseUserManager[UserModel, int]):
    """Менеджер пользователей.

    Обрабатывает регистрацию, аутентификацию и управление пользователями.
    Расширяет BaseUserManager с поддержкой int-идентификаторов.
    """
    reset_password_token_secret = config.RESET_PW_TOKEN_SECRET
    """Секретный ключ для кодирования токена сброса пароля."""

    verification_token_secret = config.VERIFICATION_TOKEN_SECRET
    """Секретный ключ для кодирования токена подтверждения
    (например email).
    """
