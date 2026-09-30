"""src.api.agent_router

Модуль роутеров FastAPI для API проекта Agent‑Assistant.
Содержит базовые эндпоинты: корневой маршрут и проверку
работоспособности сервиса (health check). Используется
для интеграции в основное приложение через включение
в экземпляр FastAPI.
"""

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.interfaces import HealthStatusProtocol, AgentProtocol, UserProtocol
from src.db import get_db
from src.health_check_service import health_check
from src.users import get_current_user
from src.api.schemas import AgentResponse, UserQuery, HealthStatus

router = APIRouter(
    tags=['agent']
)


@router.get(
    "/",
    summary='краткое описание Agent‑Assistant'
)
async def get_root() -> str:
    """Возвращает краткое описание Agent‑Assistant.

    Предоставляет базовую информацию о назначении сервиса.
    Используется как стартовый эндпоинт для проверки
    доступности API и демонстрации статуса проекта.
    """
    return "Agent-Assistant - AI‑агент для работы с внутренней документацией."


@router.get(
    "/health",
    summary='проверяет работоспособность сервиса',
    response_model=HealthStatus
)
async def get_health(request: Request) -> HealthStatusProtocol:
    """Проверяет работоспособность сервиса.

    Возвращает статус healthy при отсутствии ошибок.
    В случае исключения возвращает статус unhealthy
    с описанием ошибки. Применяется для мониторинга
    и интеграции со средствами оркестрации.
    """
    return await health_check(request)


@router.post(
    "/ask",
    summary='Обрабатывает пользовательский запрос к ИИ‑агенту',
    response_model=AgentResponse
)
async def post_user_query(
    user_query: UserQuery,
    request: Request,
    user: UserProtocol = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> AgentResponse:
    """Обрабатывает пользовательский запрос к ИИ‑агенту.

    Принимает запрос в формате UserQuery, передаёт его агенту для
    обработки в рамках RAG‑цикла (поиск по внутренней документации →
    формирование обобщённого ответа) и возвращает сформированный ответ
    в соответствии со схемой AgentResponse.
    """
    agent: AgentProtocol = request.app.state.agent
    user_id = user.id
    request_id = request.state.request_id
    try:
        answer = await agent.chat(user_id, request_id, user_query, db)
        return AgentResponse(status="success", answer=answer)
    except Exception as e:
        return AgentResponse(
            status='fail',
            answer=f"Ошибка при обработке запроса: {e}"
        )
