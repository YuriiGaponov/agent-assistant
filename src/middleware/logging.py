"""src.middleware.logging

Middleware для логирования HTTP‑запросов в проекте Agent‑Assistant.

Предоставляет:
    - Автоматическую генерацию request_id для сквозной трассировки запросов.
"""

import jwt
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import get_config
from src.logger import logger

config = get_config()


def extract_user_id(request: Request) -> str | None:
    """Достаёт user_id из JWT без зависимостей FastAPI."""
    authorization = request.headers.get("Authorization", "")
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() != "bearer" or not credentials:
        return None
    token = credentials.strip()
    try:
        payload = jwt.decode(
            token,
            config.JWT_SECRET,
            audience=["fastapi-users:auth"],
            algorithms=["HS256"],
        )
        return payload.get("sub")
    except Exception:
        return None


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware для логирования входящих HTTP‑запросов.

    Ключевые особенности:
        - Каждый запрос получает уникальный request_id (UUID).
        - В лог обработки запроса добавляется id пользователя.
    """
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        user_id = extract_user_id(request)
        try:
            data = await request.json()
        except Exception:
            # Тело запроса может отсутствовать или не быть валидным JSON
            # (например, пустой POST, multipart/form-data и т.п.).
            data = None
        with logger.contextualize(
            log_type="request",
            request_id=request_id,
            user_id=user_id
        ):
            logger.bind(step="request_start").info(
                "запрос получен",
                method=request.method,
                path=request.url.path,
                data=data
            )
            response = await call_next(request)
            logger.bind(step="request_end").info(
                "запрос выполнен",
                status_code=response.status_code,
            )
            return response
