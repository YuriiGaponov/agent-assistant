"""src.interfaces

Абстрактные классы и протоколы, определяющие контракты
различных модулей и пакетов, а также внешних систем проекта Agent‑Assistant.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, Sequence, Literal

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession


# ======= Протоколы работы с пользователями =======
class UserProtocol(Protocol):
    """Протокол модели пользователя."""
    id: int


# ======= Протоколы взаимодействия с ИИ‑агентами =======
class UserQueryProtocol(Protocol):
    """Протокол запроса пользователя, передаваемого в агент для выполнения."""
    query: str


class AgentProtocol(Protocol):
    """Протокол менеджера взаимодействия с LLM."""
    async def health_check(self) -> bool:
        """Проверка доступности LLM. Возвращает True, если LLM доступна
        должна отвечать за разумное время.
        """
        pass

    async def chat(
        self,
        user_id: int,
        request_id: str,
        query: UserQueryProtocol,
        db: AsyncSession
    ) -> str:
        """Взаимодействие с LLM в виде диалога.
        Принимает объект запроса пользователя,
        возвращает строковый ответ.
        """
        pass


class StateProtocol(Protocol):
    """
    Протокол состояния агента для ReAct/LangGraph-цикла в Agent‑Assistant.

    Должен быть структурно совместим с pydantic-моделью State из
    src.agent.state, чтобы можно было использовать её напрямую в графе
    LangGraph.

    Поля:
        - user_id, request_id: идентификаторы для трассировки и логирования
        - query: исходный запрос пользователя
        - step: текущий шаг/фаза выполнения (опционально)
        - messages: история сообщений с редуктором add_messages
        - answer: финальный/промежуточный ответ агента
    """
    user_id: int
    request_id: str | None
    query: str | None
    step: str | None
    messages: list[Any]
    answer: str


# ======= Протоколы взаимодействия с LLM =======
class LLMManagerInterface(Protocol):
    """Протокол менеджера LLM: определяет ожидаемый интерфейс для
    низкоуровневого взаимодействия с моделью:
        - привязки инструментов к модели (bind_tools),
        - вызова модели (ainvoke),
        - проверки доступности модели (health_check).

    Является внутренним контрактом пакета src.llm и используется пакетом
    src.agent (графом) для оркестрации, не зависящей от конкретного провайдера.
    """
    model: str
    """Модель, с которой работает менеджер."""

    def bind_tools(self, tools: list):
        """Привязывает инструменты к модели."""
        pass

    async def ainvoke(self, messages: list):
        """Вызывает модель с переданными сообщениями."""
        pass

    async def health_check(self) -> bool:
        """Проверяет доступность модели."""
        pass


# ======= Протоколы обработки внутренней документации =======
class SourceProtocol(Protocol):
    """Источник документа (файл, URL, база знаний и т.п.).

    Используется для отслеживания происхождения данных и дедупликации.
    """
    id: str
    path: Path
    type: str
    content_hash: str
    updated_at: datetime


class DocumentMetaProtocol(Protocol):
    """Протокол метаданных документа для RAG-пайплайна."""
    embedding_model: str
    source: SourceProtocol


class DocumentProtocol(Protocol):
    """Структура данных документа для RAG-пайплайна.

    Содержит исходный текст, список чанков и (опционально) эмбеддинги.
    Эмбеддинг может быть None на этапах чтения и чанкинга и заполняется
    позже на этапе генерации векторных представлений.
    """
    id: str
    chunk: str
    embedding: np.ndarray | None
    chunk_hash: str

    meta: DocumentMetaProtocol


# ======= Протоколы взаимодействия с векторной базой данных =======
class EmbeddingModelProtocol(Protocol):
    """Протокол модели генерации эмбеддингов"""
    def encode(
        self, chunks: list[str],
        convert_to_numpy: bool
    ) -> np.ndarray:
        pass


class VectorSearchQueryProtocol(Protocol):
    """Протокол объекта запроса для семантического поиска в векторной БД"""
    query: str


class VectorSearchResultProtocol(Protocol):
    """Протокол объекта результата семантического поиска в векторной БД,
    возвращаемого менеджером векторной БД.
    """
    documents: Sequence[str]


class VectorDBManagerInterface(Protocol):
    """Протокол менеджера векторной БД: определяет ожидаемый интерфейс для
        - создания векторной коллекции,
        - добавления документов в векторное хранилище,
        - удаления записей по источнику,
        - семантического поиска по вектору запроса,
        - корректного закрытия клиента при завершении работы.
    """
    async def create_collection(self, name: str = 'Agent-Assistant'):
        """Получает существующую или создает новую коллекцию векторной БД
        с заданным именем name.
        """
        pass

    async def add(
        self, data: DocumentMetaProtocol | list[DocumentMetaProtocol]
    ) -> None:
        """Принимает один  объект или список структуры данных Document,
        определенной в пакете RAG, содержащей данные, которые должны быть
        добавлены в векторную БД.
        """
        pass

    async def delete_by_source(self, source_id: str) -> None:
        """Удаляет записи коллекции, относящиеся к указанному источнику."""
        pass

    async def search(self, embeddings: np.ndarray):
        """Принимает преобразованный в эмбеддинги поисковый запрос
        для семантического поиска в векторной БД.
        """
        pass

    async def close(self) -> None:
        """Закрывает клиент векторной БД, корректно выгружая
        все незавершённые данные (например, HNSW-индекс) на диск.

        Вызывается при завершении работы приложения (graceful shutdown),
        чтобы избежать повреждения персистентного индекса.
        """
        pass


# ======= Протоколы взаимодействия с реляционной базой данных =======
class DBInterface(Protocol):
    """Протокол менеджера реляционной БД"""


# ======= Протоколы проверки работоспособности агента =======
class HealthStatusProtocol(Protocol):
    """Протокол состояния работоспособности систем Agent-Assistant"""
    # Общий статус
    status: Literal["healthy", "unhealthy"]
    # Доступность хранилища внутренней документации
    doc_store_available: bool
    # Доступность главной LLM.
    main_llm_available: bool
