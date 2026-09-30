"""src.db.repositories.state_repository

Репозиторий для инкапсуляции опараций с моделью состояния агента в БД.
"""

from sqlalchemy import select
from langchain_core.messages import messages_to_dict

from src.models import StateModel
from src.interfaces import StateProtocol
from src.db.repositories.base import BaseRepository


class StateRepository(BaseRepository):
    """Репозиторий для работы с сущностью состояния агента в БД."""

    async def get_by_user_id(self, user_id: int) -> StateModel | None:
        """Получает состояние из БД по id пользователя."""
        stmt = select(StateModel).where(StateModel.user_id == user_id)
        result = await self.db.execute(stmt)
        row = result.scalars().one_or_none()
        return row

    async def save(self, state: StateProtocol) -> StateModel:
        """Добавляет или обновляет состояние в сессии.

        Если записи для user_id нет в БД — создаётся новая
        и добавляется в сессию. Если есть — обновляются поля
        существующей записи. Коммит не вызывается —
        его делает сервис, управляющий транзакцией.
        """
        existing = await self.get_by_user_id(state.user_id)

        # Сообщения ReAct‑цикла — это объекты LangChain BaseMessage,
        # которые не являются JSON‑сериализуемыми. Перед записью
        # в JSON‑колонку преобразуем их в список словарей.
        serialized_messages = messages_to_dict(state.messages)

        if existing is None:
            new_state = StateModel(
                user_id=state.user_id,
                request_id=state.request_id,
                query=state.query,
                step=state.step,
                messages=serialized_messages,
                answer=state.answer,
            )
            self.db.add(new_state)
            return new_state

        existing.request_id = state.request_id
        existing.query = state.query
        existing.step = state.step
        existing.messages = serialized_messages
        existing.answer = state.answer
        return existing
