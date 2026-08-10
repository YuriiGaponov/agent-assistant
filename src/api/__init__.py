"""src.api.__init__

Пакет логики API проекта Agent‑Assistant.
Предназначен для организации импорта в основной файл main.

Состав:

Модули:
- schemas: модели валидации входящих и исходящих данных API
- routers: роутер с подключенными эндпоинтами
"""

from src.api.routers import router

__all__ = ['router']
