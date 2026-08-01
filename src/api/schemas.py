"""src.api.schemas

Модуль схем Pydantic для API проекта Agent‑Assistant.
Содержит модели данных для валидации запросов и формирования
ответов. Включает схему UserQuery для передачи пользовательского
запроса к агенту.
"""

from typing import Literal
from pydantic import BaseModel


class UserQuery(BaseModel):
    """Модель запроса пользователя к ИИ‑агенту.

    Используется для валидации входящего текста запроса.
    Поле query содержит формулировку задачи или вопроса,
    который должен обработать агент в рамках RAG‑цикла.
    """
    query: str


class AgentResponse(BaseModel):
    """Модель ответа ИИ‑агента для клиента.

    Описывает базовый формат ответа от агента. Поле status
    отражает текущее состояние обработки запроса (например,
    «success», «pending», «error»). В дальнейших версиях будет
    расширена полями с ответом агента и метаданными RAG‑цикла.
    """
    status: str


class HealthStatus(BaseModel):
    """Модель статуса работоспособности сервиса."""
    status: Literal["healthy", "unhealthy"]
    doc_store_available: bool
