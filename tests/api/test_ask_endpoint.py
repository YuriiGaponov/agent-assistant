"""tests.api.test_ask_endpoint

Модуль тестов для эндпоинта /ask проекта Agent‑Assistant.
Содержит проверки доступности маршрута и валидации формата
ответа агента. Использует тестовые данные для имитации
пользовательского запроса в рамках RAG‑сценария.
"""

from fastapi import status
from fastapi.testclient import TestClient


def test_post_ask_availible(client: TestClient, test_user_query):
    """Проверяет доступность и базовый ответ эндпоинта /ask."""

    response = client.post('/ask', json=test_user_query)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "status" in data
    assert isinstance(data["status"], str)
