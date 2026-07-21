"""tests.conftest

Модуль фикстур для тестирования Agent‑Assistant.
Фикстуры доступны во всех тестах благодаря размещению в conftest.py.
"""

import pytest
from fastapi.testclient import TestClient

from main import app
from src.api.schemas import UserQuery


@pytest.fixture
def client() -> TestClient:
    """Создаёт и возвращает тестовый клиент FastAPI.

    Фикстура предоставляет экземпляр :class:`TestClient`, привязанный
    к приложению из модуля :mod:`main`.  Благодаря размещению в
    ``conftest.py`` фикстура автоматически доступна во всех тестах
    набора.

    Returns
    -------
    TestClient
        Синхронный тестовый клиент, сконфигурированный с основным
        FastAPI-приложением.
    """
    return TestClient(app)


@pytest.fixture
def test_user_query() -> dict:
    """Возвращает тестовые данные для запроса к ИИ‑агенту.

    Генерирует словарь на основе модели UserQuery с фиксированным
    текстом запроса. Используется в тестах для имитации входящих
    данных от пользователя в рамках RAG‑сценариев.
    """
    return UserQuery(query='Тестовый запрос пользователя').model_dump()
