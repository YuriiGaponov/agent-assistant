"""src.middleware.__init__

Пакета middleware для проекта Agent‑Assistant.

Включает:
- logging.py: логирование запросов пользователей
"""

from src.middleware.logging import RequestLoggingMiddleware

__all__ = ['RequestLoggingMiddleware']
