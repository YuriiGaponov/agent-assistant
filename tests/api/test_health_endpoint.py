"""tests.api.test_health_endpoint.py

Модуль тестов для проверки работоспособности сервиса.
Содержит тесты базовых эндпоинтов API проекта Agent‑Assistant.
В текущей версии реализует проверку health‑check через
запрос к маршруту /health.
"""

from fastapi import status
from fastapi.testclient import TestClient


def test_get_health(client: TestClient) -> None:
    """Проверяет доступность health‑эндпоинта.

    Отправляет GET‑запрос на маршрут /health и убеждается,
    что сервис возвращает статус 200 (OK). Подтверждает
    базовую работоспособность API и корректность обработки
    запросов в рамках тестового клиента.
    """
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
