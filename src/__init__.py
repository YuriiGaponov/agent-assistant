"""
src.__init__

Корневой пакет проекта Agent‑Assistant.
Предназначен для организации импорта в основной файл main.

Состав:

Модули:
- config: конфигурация проекта
- logger: механизм логирования

Пакеты:
- api: реализация эндпоинтов взаимодействия с пользователем
"""

from src.api import router
from src.config import get_config
from src.logger import logger

__all__ = ['get_config', 'logger', 'router']
