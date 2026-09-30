"""tests.conftest

Модуль фикстур для тестирования Agent‑Assistant.
Фикстуры доступны во всех тестах благодаря размещению в conftest.py.
"""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from main import app
from src.config import Config
from src.api.schemas import UserQuery


@pytest.fixture
def mock_agent():
    """Возвращает мок‑объект агента с асинхронными методами chat() и health_check().

    Мок заменяет реальный объект агента, что позволяет тестировать эндпоинты
    в изоляции от внешних зависимостей (Ollama, векторная БД).

    Returns:
        AsyncMock: Мок‑объект агента с предопределёнными ответами.
    """
    agent = AsyncMock()
    agent.chat = AsyncMock(return_value="Тестовый ответ агента")
    agent.health_check = AsyncMock(return_value=True)
    return agent


@pytest.fixture
def client(mock_agent, test_config) -> TestClient:
    """Создаёт и возвращает тестовый клиент FastAPI.

    Фикстура предоставляет экземпляр :class:`TestClient`, привязанный
    к приложению из модуля :mod:`main`.  Устанавливает в ``app.state``
    мок‑объекты (agent, config, embedding_model, vectorDB, tools), чтобы
    тесты были изолированы от внешних зависимостей и не требовали запуска
    lifespan‑контекста.

    Returns
    -------
    TestClient
        Синхронный тестовый клиент, сконфигурированный с основным
        FastAPI-приложением и мок‑зависимостями.
    """
    app.state.config = test_config
    app.state.embedding_model = AsyncMock()
    app.state.vectorDB = AsyncMock()
    app.state.tools = AsyncMock()
    app.state.agent = mock_agent
    return TestClient(app)


@pytest.fixture
def test_config() -> Config:
    """Возвращает экземпляр конфигурации для тестов.

    Позволяет переопределять отдельные поля (например, пути, флаги)
    в параметризованных тестах без изменения глобальной конфигурации.

    Returns:
        Config: Настроенный объект конфигурации.
    """
    return Config()


@pytest.fixture
def test_user_query() -> dict:
    """Возвращает тестовые данные для запроса к ИИ‑агенту.

    Генерирует словарь на основе модели UserQuery с фиксированным
    текстом запроса. Используется в тестах для имитации входящих
    данных от пользователя в рамках RAG‑сценариев.
    """
    return UserQuery(query='Тестовый запрос пользователя').model_dump()