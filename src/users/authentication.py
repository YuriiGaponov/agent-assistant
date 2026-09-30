"""src.users.authentication

Модуль аутентификации пользователей.
"""

from fastapi_users.authentication import (
    AuthenticationBackend, BearerTransport, JWTStrategy
)

from src.config import get_config

config = get_config()

bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")
"""Трансаорт токена аутентификации."""


def get_jwt_strategy() -> JWTStrategy:
    """Возвращает экземпляр JWT-стратегии с секретом из конфигурации."""
    return JWTStrategy(secret=config.JWT_SECRET, lifetime_seconds=3600)


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)
"""Бэкенд аутентификации."""
