"""src.api.routers

Модуль роутеров FastAPI для API проекта Agent‑Assistant.
Содержит базовые эндпоинты: корневой маршрут и проверку
работоспособности сервиса (health check). Используется
для интеграции в основное приложение через включение
в экземпляр FastAPI.
"""

from fastapi import APIRouter, Request

from src.interfaces import HealthStatusProtocol, AgentProtocol
from src.health_check_service import health_check
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
async def post_user_query(
    user_query: UserQuery,
    request: Request
) -> AgentResponse:
    """Обрабатывает пользовательский запрос к ИИ‑агенту.

    Принимает запрос в формате UserQuery, передаёт его агенту для
    обработки в рамках RAG‑цикла (поиск по внутренней документации →
    формирование обобщённого ответа) и возвращает сформированный ответ
    в соответствии со схемой AgentResponse.
    """
    agent: AgentProtocol = request.app.state.agent
    answer = await agent.chat(user_query)
    return AgentResponse(status="success", answer=answer)
