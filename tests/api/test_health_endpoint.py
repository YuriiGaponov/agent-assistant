"""tests.api.test_health_endpoint

Модуль тестов для проверки работоспособности сервиса.
Содержит тесты базовых эндпоинтов API проекта Agent‑Assistant.
В текущей версии реализует проверку health‑check через
запрос к маршруту /health.
"""

from fastapi import status
from fastapi.testclient import TestClient

from src.api.schemas import HealthStatus


def test_get_health(client: TestClient) -> None:
    """Проверяет доступность health‑эндпоинта и корректность ответа.

    Тест убеждается, что:
      - возвращается HTTP 200;
      - JSON валидируется схемой HealthStatus;
      - поля имеют ожидаемые типы и значения (минимальная семантика).
    """
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    HealthStatus(**data)
