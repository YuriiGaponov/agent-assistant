"""tests.api.test_root_endpoint"""

from fastapi import status
from fastapi.testclient import TestClient


def test_get_root(client: TestClient):
    """Тест корневого эндпоинта.

    Отправляет GET‑запрос на корневой URL ('/') и проверяет,
    что код состояния ответа равен 200 (OK).

    Args:
        client: фикстура тестового клиента FastAPI.

    Asserts:
        response.status_code == 200
    """
    response = client.get('/')
    assert response.status_code == status.HTTP_200_OK
