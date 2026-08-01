"""src.api.routers

Модуль роутеров FastAPI для API проекта Agent‑Assistant.
Содержит базовые эндпоинты: корневой маршрут и проверку
работоспособности сервиса (health check). Используется
для интеграции в основное приложение через включение
в экземпляр FastAPI.
"""

from fastapi import APIRouter, Request

from src.interfaces import HealthStatusProtocol
from src.health_check_service import health_check
from src.RAG import semantic_search
from src.api.schemas import AgentResponse, UserQuery, HealthStatus

router = APIRouter()


@router.get("/")
async def get_root() -> str:
    """Возвращает краткое описание Agent‑Assistant.

    Предоставляет базовую информацию о назначении сервиса.
    Используется как стартовый эндпоинт для проверки
    доступности API и демонстрации статуса проекта.
    """
    return "Agent-Assistant - AI‑агент для работы с внутренней документацией."


@router.get("/health", response_model=HealthStatus)
async def get_health(request: Request) -> HealthStatusProtocol:
    """Проверяет работоспособность сервиса.

    Возвращает статус healthy при отсутствии ошибок.
    В случае исключения возвращает статус unhealthy
    с описанием ошибки. Применяется для мониторинга
    и интеграции со средствами оркестрации.
    """
    return await health_check(request)


@router.post("/ask", response_model=AgentResponse)
async def post_user_query(user_query: UserQuery):
    """Обрабатывает пользовательский запрос к ИИ‑агенту.

    Принимает запрос в формате UserQuery и возвращает ответ
    в соответствии со схемой AgentResponse. На текущем этапе
    возвращает статус success; в дальнейшем будет расширен
    логикой RAG‑цикла и формированием содержательного ответа.
    """
    return {"status": "success"}


# Тестовый эндпоинт, в релизе будет удален
@router.post("/test_query")
async def test_post_user_query(
    user_query: UserQuery,
    request: Request
):
    """Тестовый эндпоинт"""
    # embedding_model = request.app.state.embedding_model
    result = await semantic_search(user_query, request)
    return result
