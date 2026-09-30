"""src.agent.services"""

from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import messages_from_dict

from src.db import StateRepository
from src.models import StateModel
from src.agent.state import State


async def get_state_by_user_id(
    user_id: int,
    session: AsyncSession
) -> State:
    """
    Получает состояние агента для пользователя из БД.

    Если запись найдена — конвертирует StateModel -> State.
    Если не найдена — возвращает новый пустой State с user_id.

    Args:
        user_id: идентификатор пользователя.
        session: асинхронная сессия SQLAlchemy.

    Returns:
        Объект State (либо восстановленный из БД, либо новый).
    """
    repo = StateRepository(session)
    state_db: StateModel | None = await repo.get_by_user_id(user_id)

    if state_db is not None:
        return State(
            user_id=state_db.user_id,
            request_id=state_db.request_id,
            query=state_db.query,
            step=state_db.step,
            # В БД сообщения хранятся в виде списка словарей (JSON).
            # Восстанавливаем из них объекты LangChain BaseMessage.
            messages=messages_from_dict(state_db.messages),
            answer=state_db.answer,
        )

    return State(user_id=user_id)


async def save_state(
    state: State,
    session: AsyncSession
) -> None:
    """Сохраняет состояние агента в БД и коммитит транзакцию.
    Очищает историю диалога с моделью.
    """
    repo = StateRepository(session)

    state.messages = []
    await repo.save(state)
    await session.commit()
