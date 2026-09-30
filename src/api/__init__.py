"""src.api.__init__

Пакет логики API проекта Agent‑Assistant.
Предназначен для организации импорта в основной файл main.

Состав:

Модули:
- schemas: модели валидации входящих и исходящих данных API
- agent_router: роутер с подключенными эндпоинтами агента
- auth_router: роутер аутентификации
- user_router: роутер для работы с пользователями
  (регистрация, аутентификация и т.д.)
"""

from fastapi import APIRouter

from src.api.agent_router import router as agent_router
from src.api.auth_router import router as auth_router
from src.api.user_router import router as user_router

router = APIRouter()

router.include_router(agent_router)
router.include_router(auth_router)
router.include_router(user_router)

__all__ = ['router']
