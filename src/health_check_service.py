"""src.health_check_service

Сервис проверки работоспособности Agent‑Assistant
"""

from dataclasses import dataclass
from typing import Literal

from fastapi import Request

from src.interfaces import HealthStatusProtocol, AgentProtocol
from src.RAG import doc_dir_exist


@dataclass
class HealthStatus:
    """Модель статуса работоспособности сервиса."""
    status: Literal["healthy", "unhealthy"] = "healthy"
    doc_store_available: bool = True
    main_llm_available: bool = True


async def health_check(request: Request) -> HealthStatusProtocol:
    """Проверяет работоспособность основных компонентов Agent‑Assistant.

    Выполняет следующие проверки:
      1. Доступность директории с документацией (doc_dir_exist).
      2. Доступность главной LLM (mainLLM.health_check()).

    Args:
        request: FastAPI Request, через который получаем доступ к app.state.

    Returns:
        HealthStatusProtocol: Объект со статусом и деталями проверок.
    """
    status = HealthStatus()
    config = request.app.state.config
    # Доступность хранилища внутренней документации
    doc_store_available = doc_dir_exist(config)
    if not doc_store_available:
        status.status = "unhealthy"
        status.doc_store_available = doc_store_available
    # Доступность главной LLM.
    mainLLM: AgentProtocol = request.app.state.mainLLM
    main_llm_available = await mainLLM.health_check()
    if not main_llm_available:
        status.status = "unhealthy"
        status.main_llm_available = main_llm_available

    return status
