"""src.models.state

Модуль ORM‑модели состояния агента для БД.

StateModel зеркалирует поля Pydantic‑модели State из src.agent.state
и позволяет сохранять/восстанавливать состояние графа между запусками.
"""

from typing import Any

from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base


class StateModel(Base):
    """ORM‑модель состояния агента, соответствующая State из src.agent.state.

    Используется для персистентного хранения состояния LangGraph‑графа:
    позволяет сохранить промежуточный результат, восстановить его при
    перезапуске сервиса или проанализировать после выполнения.

    Поля:
        id — внутренний первичный ключ.
        user_id — идентификатор пользователя (соответствует State.user_id).
        user_id — идентификатор пользователя (FK на users.id).
        query — исходный запрос пользователя (соответствует State.query).
        step — текущий шаг агента (соответствует State.step).
        messages — сериализованная история сообщений ReAct‑цикла
            (соответствует State.messages; хранится в JSON‑колонке).
        answer — итоговый ответ агента (соответствует State.answer).
    """

    __tablename__ = "agent_state"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    """Идентификатор пользователя (связь с таблицей users)."""

    request_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True, index=True
    )
    """Идентификатор запроса пользователя."""

    query: Mapped[str | None] = mapped_column(Text, nullable=True)
    """Запрос пользователя."""

    step: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None
    )
    """Текущий шаг работы агента."""

    messages: Mapped[list[Any]] = mapped_column(
        JSON, nullable=False, default=list
    )
    """История сообщений ReAct‑цикла в виде списка словарей.
    Хранится как JSON, поскольку LangGraph messages — это список
    объектов BaseMessage, которые сериализуются в dict."""

    answer: Mapped[str] = mapped_column(Text, nullable=False, default="")
    """Итоговый ответ агента."""
