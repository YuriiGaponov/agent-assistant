"""src.models.base

Базовый класс для всех SQLAlchemy-моделей проекта Agent‑Assistant.
"""

from sqlalchemy import Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Базовый класс-основа для всех ORM‑моделей в проекте."""

    id: Mapped[int] = mapped_column(
            Integer, primary_key=True, autoincrement=True
        )
    """Внутренний идентификатор записи."""
