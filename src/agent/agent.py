"""src.agent.agent

Реализация агента Agent‑Assistant.

Класс Agent инкапсулирует всю логику оркестрации RAG‑цикла без привязки
к конкретной реализации LLM‑менеджера:
    - создаёт менеджер LLM через фабрику src.llm.init_llm (реализующий
      интерфейс LLMManagerInterface);
    - привязывает инструменты к менеджеру;
    - строит граф LangGraph (src.agent.graph.build_graph);
    - предоставляет методы init_state, health_check, chat.

Вся низкоуровневая логика взаимодействия с моделью (создание клиента,
вызов, проверка доступности) делегируется менеджеру LLM из пакета src.llm.
Агент не знает, с каким провайдером (Ollama, Mistral и т.д.) работает, —
он взаимодействует только через интерфейс LLMManagerInterface.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from src.config import Config
from src.logger import logger
from src.interfaces import (
    EmbeddingModelProtocol, LLMManagerInterface,
    UserQueryProtocol, VectorDBManagerInterface
)
from src.agent.state import State
from src.agent.graph import build_graph
from src.agent.services import get_state_by_user_id, save_state
from src.agent.tools import make_tools
from src.llm import init_llm


class Agent:
    """Агент, оркестрирующий RAG‑цикл через граф LangGraph.

    Инкапсулирует создание LLM‑менеджера, привязку инструментов,
    построение графа и взаимодействие с моделью через LLMManagerInterface.
    Не зависит от конкретной реализации менеджера LLM.
    """

    def __init__(
        self,
        config: Config,
        embedding_model: EmbeddingModelProtocol,
        vector_db: VectorDBManagerInterface,
    ):
        self.llm_manager: LLMManagerInterface = init_llm(config)
        tools = make_tools(embedding_model, vector_db)
        self.llm_manager.bind_tools(tools)
        self.graph = build_graph(self.llm_manager, tools)
        logger.debug(
            "инициализирован агент",
            agent=Agent,
            model=self.llm_manager.model
        )

    async def init_state(
        self,
        user_id: int,
        request_id: str,
        query: UserQueryProtocol,
        db: AsyncSession
    ) -> State:
        """Инициализирует начальное состояние агента для RAG‑цикла.

        Получает сохранённое состояние пользователя из БД через
        get_state_by_user_id, затем обновляет его поля request_id
        и query актуальными значениями. Реализация асинхронна,
        поскольку предполагает обращение к реляционной БД.

        Args:
            user_id (int): Идентификатор пользователя для загрузки
                состояния.
            request_id (str): Уникальный идентификатор текущего
                запроса для трассировки.
            query (UserQueryProtocol): Объект запроса пользователя,
                содержащий текст запроса.
            db (AsyncSession): Асинхронная сессия SQLAlchemy для
                взаимодействия с БД.

        Returns:
            State: Инициализированный объект состояния агента,
                готовый к передаче в граф LangGraph.

        Raises:
            Exception: Пробрасывается любая ошибка, возникшая
                при получении или обновлении состояния.
        """
        state = await get_state_by_user_id(user_id, db)
        state.request_id = request_id
        state.query = query.query
        return state

    async def health_check(self) -> bool:
        """Проверяет доступность модели через менеджер LLM.

        Делегирует проверку конкретному менеджеру (src.llm), который
        знает, как проверить доступность своего провайдера.

        Returns:
            bool: True, если модель доступна и отвечает; иначе False.
        """
        return await self.llm_manager.health_check()

    async def chat(
        self,
        user_id: int,
        request_id: str,
        query: UserQueryProtocol,
        db: AsyncSession
    ) -> str:
        """Обрабатывает пользовательский запрос через граф RAG‑цикла.

        Инициализирует начальное состояние, запускает граф агента
        (цикл «вызов модели → при необходимости вызов инструментов»)
        и возвращает итоговый обобщённый ответ.

        Args:
            query: Объект запроса пользователя (UserQueryProtocol).

        Returns:
            str: Обобщённый ответ агента.
        """
        logger.info('получение initial_state', step='initial_state')
        try:
            initial_state = await self.init_state(
                user_id, request_id, query, db
            )
            logger.info(
                'initial_state получен',
                step='initial_state',
                initial_state=initial_state
            )
        except Exception as e:
            logger.error(
                'ошибка получения initial_state',
                step='initial_state',
                error=e
            )
            raise
        logger.info('передача initial_state в граф')
        try:
            result = await self.graph.ainvoke(initial_state)
            logger.info('получен результат выполнения графа', result=result)
            result_state = State.model_validate(result)
            await save_state(result_state, db)
            return result_state.answer
        except Exception as e:
            logger.error('ошибка выполнения графа', error=e)
            raise
